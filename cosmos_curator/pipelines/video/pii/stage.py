# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Curator stage that blurs faces in each clip after transcode.

The stage is off unless the split pipeline is asked for it. On success it
replaces ``clip.encoded_data`` with the redacted MP4. On failure it records
``clip.errors["pii_redaction"]`` and moves the clip to ``filtered_clips``.

That move does not hold the bytes back. The clip writer still uploads
``filtered_clips`` as MP4. Dropping the bytes on failure is a later change.
"""

from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.pipelines.video.pii.builders import PiiRedactionStageConfig
from cosmos_curator.pipelines.video.pii.config import EncodeConfig, PaintConfig, RedactionConfig
from cosmos_curator.pipelines.video.pii.detector import PiiDetector
from cosmos_curator.pipelines.video.pii.redact import redact_mp4
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video


class PiiRedactionStage(CuratorStage):
    """Blur faces in every clip, then hand the redacted MP4 to the next stage."""

    def __init__(self, config: PiiRedactionStageConfig, detector: PiiDetector | None = None) -> None:
        """Store the paint settings. The model loads in :meth:`stage_setup` unless a detector is injected."""
        self._config = config
        self._detector = detector
        self._redaction = RedactionConfig(
            paint=PaintConfig(
                mode=config.paint_mode,
                mask_growth=config.mask_growth,
                missed_look_growth=config.missed_look_growth,
                blur_kernel=config.blur_kernel,
                blur_downscale=config.blur_downscale,
                pixelate_block=config.pixelate_block,
            ),
            encode=EncodeConfig(),
        )

    @property
    def resources(self) -> CuratorStageResource:
        """One GPU. The still-image SAM3 detector runs on CUDA."""
        return CuratorStageResource(cpus=1.0, gpus=self._config.gpus_per_worker)

    def stage_setup(self) -> None:
        """Load ``facebook/sam3`` when no detector was injected."""
        if self._detector is None:
            from cosmos_curator.pipelines.video.pii.sam3_image import Sam3ImageDetector  # noqa: PLC0415

            detector = Sam3ImageDetector()
            detector.setup()
            self._detector = detector

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        """Redact each clip. A failed clip moves to ``filtered_clips`` with its bytes intact."""
        if self._detector is None:
            msg = "stage_setup() must run before process_data()"
            raise RuntimeError(msg)
        for task in tasks:
            for video in task.videos:
                self._redact_video(video)
        return tasks

    def _redact_video(self, video: Video) -> None:
        kept: list[Clip] = []
        for clip in video.clips:
            try:
                self._redact_clip(clip)
            except Exception:  # noqa: BLE001 -- one clip must not drop the rest of the batch
                logger.exception(f"PII redaction failed for clip {clip.uuid} in {video.input_video}")
                clip.errors["pii_redaction"] = "redaction failed"
                video.filtered_clips.append(clip)
            else:
                kept.append(clip)
        video.clips = kept
        if self._config.verbose:
            logger.info(
                f"Video {video.input_video} chunk-{video.clip_chunk_index} has "
                f"{len(video.clips)}/{len(video.filtered_clips)} clips passed/filtered by PII redaction"
            )

    def _redact_clip(self, clip: Clip) -> None:
        raw = clip.encoded_data.resolve()
        if raw is None or len(raw) == 0:
            msg = f"clip {clip.uuid} has no encoded video"
            raise ValueError(msg)
        assert self._detector is not None
        result = redact_mp4(raw.tobytes(), self._detector, self._redaction)
        clip.encoded_data = bytes_to_numpy(result.mp4_bytes)  # type: ignore[assignment]
        clip.pii_mask_archive = result.mask_archive
