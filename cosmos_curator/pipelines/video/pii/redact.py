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

"""Redact one clip: decode once, track, paint, encode once.

This is the whole engine as one function over MP4 bytes. It returns new bytes
and leaves the input alone, so the caller decides what to replace and only does
so after this returns. If anything fails, it raises and the caller still holds
the original, never a half-written clip.
"""

from collections.abc import Iterable, Iterator

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.pipelines.video.pii.config import PaintMode, RedactionConfig
from cosmos_curator.pipelines.video.pii.decode import FullRateDecoder
from cosmos_curator.pipelines.video.pii.detector import PiiDetector
from cosmos_curator.pipelines.video.pii.encode import (
    RedactionError,
    encode_frames,
    nominal_frame_rate,
    resolve_bitrate,
    verify_encoded,
)
from cosmos_curator.pipelines.video.pii.mask_archive import MaskArchiveWriter
from cosmos_curator.pipelines.video.pii.paint import paint_frame, union_mask
from cosmos_curator.pipelines.video.pii.prefetch import PrefetchingDetector
from cosmos_curator.pipelines.video.pii.propagate import LKRigidPropagator, MaskPropagator
from cosmos_curator.pipelines.video.pii.tracker import FrameResult, PiiTracker, TrackerStats


@attrs.define(eq=False)
class RedactionResult:
    """What a successful redaction produced.

    Attributes:
        mp4_bytes: the redacted clip, same frames and timestamps as the input.
        mask_archive: the compressed mask archive for ``none`` mode, ``None`` for ``gaussian``.
        stats: tracker counters for audit.
        frame_count: frames in and out.
        timestamps_ns: per-frame presentation time, relative to the first frame.

    """

    mp4_bytes: bytes
    mask_archive: bytes | None
    stats: TrackerStats
    frame_count: int
    timestamps_ns: npt.NDArray[np.int64]


def redact_mp4(
    mp4_bytes: bytes,
    detector: PiiDetector,
    config: RedactionConfig,
    propagator: MaskPropagator | None = None,
) -> RedactionResult:
    """Redact every face the detector finds in ``mp4_bytes``.

    Raises:
        RedactionError: the output could not be verified against the input.
        ValueError: the clip cannot be encoded as written (for example odd dimensions).

    """
    decoder = FullRateDecoder(mp4_bytes)
    paint = config.paint
    prefetch: PrefetchingDetector | None = None
    if config.detector_lookahead_frames > 0:
        prefetch = PrefetchingDetector(detector, config.tracker.look_interval, config.detector_lookahead_frames)
        detector = prefetch
    tracker = PiiTracker(detector, config.tracker, propagator or LKRigidPropagator(config.lk))

    archive: MaskArchiveWriter | None = None
    if paint.mode is PaintMode.NONE:
        archive = MaskArchiveWriter(
            decoder.width,
            decoder.height,
            decoder.timestamps_ns,
            info={
                "paint_mode": paint.mode.value,
                "mask_growth": paint.mask_growth,
                "missed_look_growth": paint.missed_look_growth,
            },
        )

    def painted() -> Iterator[npt.NDArray[np.uint8]]:
        def handle(results: list[FrameResult]) -> Iterator[npt.NDArray[np.uint8]]:
            for result in results:
                mask = union_mask(result.regions, paint)
                if archive is not None:
                    archive.add(result.index, mask)
                yield paint_frame(result.frame_rgb, mask, paint)

        frames: Iterable[npt.NDArray[np.uint8]] = decoder.frames(config.decode_window_frames)
        if prefetch is not None:
            frames = prefetch.frames(frames)
        for frame in frames:
            yield from handle(tracker.push(frame))
        yield from handle(tracker.finish())

    frame_rate = nominal_frame_rate(decoder.avg_frame_rate, decoder.timestamps_ns)
    try:
        out_bytes = encode_frames(
            painted(),
            decoder.timestamps_ns,
            width=decoder.width,
            height=decoder.height,
            frame_rate=frame_rate,
            bitrate_bps=resolve_bitrate(config.encode, decoder.bit_rate_bps, len(mp4_bytes), frame_rate),
            config=config.encode,
        )
    finally:
        if prefetch is not None:
            prefetch.close()
    verify_encoded(out_bytes, decoder.timestamps_ns, width=decoder.width, height=decoder.height)

    return RedactionResult(
        mp4_bytes=out_bytes,
        mask_archive=archive.finish() if archive is not None else None,
        stats=tracker.stats,
        frame_count=decoder.frame_count,
        timestamps_ns=decoder.timestamps_ns,
    )


__all__ = ["RedactionError", "RedactionResult", "redact_mp4"]
