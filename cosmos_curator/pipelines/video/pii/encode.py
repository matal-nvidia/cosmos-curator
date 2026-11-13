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

"""Encode redacted frames back to an MP4 at the clip's own timestamps.

Every frame is written with the presentation time it was decoded with, so a
variable-frame-rate clip stays variable-frame-rate and frame ``i`` of the output
is frame ``i`` of the input. The output is then re-opened and checked against the
timestamps before it is allowed to replace anything.
"""

import io
import itertools
from collections.abc import Iterable
from fractions import Fraction
from typing import cast

import av
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.pipelines.video.pii.config import EncodeConfig

# MPEG's 90 kHz clock: exact for 24, 25, 30, 50 and 60 fps and for the 1001 variants.
TIME_BASE = Fraction(1, 90_000)
_NS_PER_SECOND = 1_000_000_000
_TICK_NS = _NS_PER_SECOND // 90_000 + 1  # one clock tick, rounded up, is the timestamp tolerance
_PIXEL_FORMAT = "yuv420p"


class RedactionError(RuntimeError):
    """The redacted clip cannot be trusted, so it must not replace the original."""


def ns_to_ticks(timestamp_ns: int) -> int:
    """Nanoseconds to 90 kHz ticks, rounded to nearest."""
    return (timestamp_ns * 90_000 + _NS_PER_SECOND // 2) // _NS_PER_SECOND


def nominal_frame_rate(avg_frame_rate: Fraction, timestamps_ns: npt.NDArray[np.int64]) -> Fraction:
    """Return the container's rate, or one derived from the timestamps if it declares none."""
    if avg_frame_rate > 0:
        return avg_frame_rate
    if len(timestamps_ns) < 2:  # noqa: PLR2004
        return Fraction(1)
    span_s = Fraction(int(timestamps_ns[-1] - timestamps_ns[0]), _NS_PER_SECOND)
    return Fraction(len(timestamps_ns) - 1) / span_s


def resolve_bitrate(config: EncodeConfig, input_bit_rate_bps: int, input_size_bytes: int, frame_rate: Fraction) -> int:
    """Return the configured bitrate, else the input's measured one, else one estimated from its size.

    The size estimate treats the clip as ``frame_rate`` frames per second of one frame each.
    """
    if config.bitrate_bps is not None:
        return config.bitrate_bps
    if input_bit_rate_bps > 0:
        return input_bit_rate_bps
    return max(1, int(input_size_bytes * 8 * frame_rate))


def encode_frames(  # noqa: PLR0913 -- these are the encode's inputs, not options to group
    frames: Iterable[npt.NDArray[np.uint8]],
    timestamps_ns: npt.NDArray[np.int64],
    *,
    width: int,
    height: int,
    frame_rate: Fraction,
    bitrate_bps: int,
    config: EncodeConfig,
) -> bytes:
    """Encode ``frames`` to MP4 bytes, frame ``i`` at ``timestamps_ns[i]``.

    Raises:
        ValueError: odd dimensions (``yuv420p`` needs even ones), timestamps that
            are not strictly increasing at 90 kHz, or a frame whose shape is wrong.
        RedactionError: the number of frames is not the number of timestamps.

    """
    if width % 2 or height % 2:
        msg = f"yuv420p needs even dimensions, got {width}x{height}"
        raise ValueError(msg)
    ticks = [ns_to_ticks(int(t)) for t in timestamps_ns]
    if any(b <= a for a, b in itertools.pairwise(ticks)):
        msg = "timestamps are not strictly increasing at 90 kHz"
        raise ValueError(msg)

    buffer = io.BytesIO()
    written = 0
    with av.open(buffer, mode="w", format="mp4") as container:
        # add_stream widens to a Video|Audio|Subtitle union for a non-literal encoder name;
        # only a video encoder is ever requested here.
        stream = cast("av.video.stream.VideoStream", container.add_stream(config.encoder, rate=frame_rate))
        stream.width = width
        stream.height = height
        stream.pix_fmt = _PIXEL_FORMAT
        stream.bit_rate = bitrate_bps
        stream.codec_context.time_base = TIME_BASE
        for frame in frames:
            if frame.shape != (height, width, 3):
                msg = f"frame {written} has shape {frame.shape}, expected {(height, width, 3)}"
                raise ValueError(msg)
            if written >= len(ticks):
                msg = f"more frames than the {len(ticks)} timestamps"
                raise RedactionError(msg)
            video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
            video_frame.pts = ticks[written]
            video_frame.time_base = TIME_BASE
            container.mux(stream.encode(video_frame))
            written += 1
        container.mux(stream.encode(None))
    if written != len(ticks):
        msg = f"encoded {written} frames but expected {len(ticks)}"
        raise RedactionError(msg)
    return buffer.getvalue()


def verify_encoded(mp4_bytes: bytes, timestamps_ns: npt.NDArray[np.int64], *, width: int, height: int) -> None:
    """Open the output and check frame count, geometry and timestamps against the input's.

    This reads the container index only; no frame is decoded.

    Raises:
        RedactionError: on any mismatch.

    """
    try:
        sensor = CameraSensor(mp4_bytes)
    except Exception as exc:
        msg = f"the encoded clip cannot be opened: {exc}"
        raise RedactionError(msg) from exc
    metadata = sensor.video_metadata
    if (metadata.width, metadata.height) != (width, height):
        msg = f"output is {metadata.width}x{metadata.height}, expected {width}x{height}"
        raise RedactionError(msg)
    out_ns = np.asarray(sensor.timestamps_ns, dtype=np.int64)
    if len(out_ns) != len(timestamps_ns):
        msg = f"output has {len(out_ns)} frames, expected {len(timestamps_ns)}"
        raise RedactionError(msg)
    drift = np.abs((out_ns - out_ns[0]) - timestamps_ns)
    if int(drift.max()) > _TICK_NS:
        msg = f"output timestamps drift by up to {int(drift.max())} ns from the input's"
        raise RedactionError(msg)
