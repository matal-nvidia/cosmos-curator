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

"""The Curator stage around redact_mp4. The detector is injected, so these tests load no model."""

import pathlib
import uuid

import numpy as np

from cosmos_curator.pipelines.video.pii.builders import PiiRedactionStageConfig
from cosmos_curator.pipelines.video.pii.config import PaintMode
from cosmos_curator.pipelines.video.pii.decode import FullRateDecoder
from cosmos_curator.pipelines.video.pii.stage import PiiRedactionStage
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video
from tests.cosmos_curator.pipelines.video.pii._pii_test_utils import (
    Scene,
    ScriptedDetector,
    disc_mask,
    encode_test_clip,
    face,
)

_GROWTH = 1.25
_MISSED = 1.5
_KERNEL = 31


def _config(mode: PaintMode = PaintMode.GAUSSIAN) -> PiiRedactionStageConfig:
    return PiiRedactionStageConfig(
        mask_growth=_GROWTH,
        missed_look_growth=_MISSED,
        blur_kernel=_KERNEL if mode is PaintMode.GAUSSIAN else None,
        paint_mode=mode,
    )


def _clip_bytes() -> bytes:
    scene = Scene()
    frames = [scene.render((120, 120)) for _ in range(8)]
    return encode_test_clip(frames)


def _task(*payloads: bytes | None) -> SplitPipeTask:
    clips = [
        Clip(
            uuid=uuid.uuid4(),
            source_video="test.mp4",
            span=(0.0, 1.0),
            encoded_data=payload,
        )
        for payload in payloads
    ]
    return SplitPipeTask(
        session_id="test-session",
        video=Video(input_video=pathlib.Path("test.mp4"), clips=clips),
    )


def _detector() -> ScriptedDetector:
    mask = disc_mask((120, 120))
    return ScriptedDetector(lambda index: [face(mask)] if index % 5 == 0 or index == 7 else [])


def _bytes(clip: Clip) -> bytes:
    raw = clip.encoded_data.resolve()
    assert raw is not None
    return raw.tobytes()


def test_a_successful_clip_is_replaced() -> None:
    """The redacted MP4 replaces the clip, and the clip stays on the passing list."""
    original = _clip_bytes()
    task = _task(original)
    stage = PiiRedactionStage(_config(), _detector())

    stage.process_data([task])

    clip = task.video.clips[0]
    assert task.video.filtered_clips == []
    assert _bytes(clip) != original
    assert FullRateDecoder(_bytes(clip)).frame_count == 8
    assert clip.pii_mask_archive is None
    assert "pii_redaction" not in clip.errors


def test_none_mode_keeps_a_mask_archive_on_the_clip() -> None:
    """None mode does not blur, and the archive stays on the clip for a later writer."""
    task = _task(_clip_bytes())
    stage = PiiRedactionStage(_config(PaintMode.NONE), _detector())

    stage.process_data([task])

    clip = task.video.clips[0]
    assert clip.pii_mask_archive is not None
    assert clip.pii_mask_archive[:2] == b"PK"


def test_a_failed_clip_moves_to_filtered_clips_with_its_bytes() -> None:
    """Failure records an error and moves the clip. The original bytes stay, and are still uploadable."""
    original = _clip_bytes()
    task = _task(original)

    class _Boom:
        def detect(self, frame_rgb: np.ndarray, frame_index: int) -> list:  # noqa: ARG002
            msg = "detector down"
            raise RuntimeError(msg)

    stage = PiiRedactionStage(_config(), _Boom())  # type: ignore[arg-type]
    stage.process_data([task])

    assert task.video.clips == []
    clip = task.video.filtered_clips[0]
    assert clip.errors["pii_redaction"] == "redaction failed"
    assert _bytes(clip) == original


def test_one_failure_does_not_drop_the_other_clip() -> None:
    """A clip with no bytes is filtered. The clip beside it is still redacted."""
    task = _task(b"", _clip_bytes())
    stage = PiiRedactionStage(_config(), _detector())

    stage.process_data([task])

    assert len(task.video.clips) == 1
    assert len(task.video.filtered_clips) == 1
    assert task.video.filtered_clips[0].errors["pii_redaction"] == "redaction failed"
    assert FullRateDecoder(_bytes(task.video.clips[0])).frame_count == 8
