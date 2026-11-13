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

"""Full-rate decode: every displayable frame of a clip, each with its real timestamp.

``decode_clip_at_fps`` (``tracking/sensor_decode.py``) builds a regular grid at a
target rate and snaps it to real frames, so it drops and duplicates frames. PII
redaction must see and rewrite every frame, so this module asks the sensor
library for the clip's own timestamps instead of a grid.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor

if TYPE_CHECKING:
    from fractions import Fraction

# Asking for a frame at exactly its own timestamp selects that frame, so the
# nearest-timestamp policy is exact here and never needs a tolerance.
_NEAREST = NearestTimestampPolicy(max_delta_ns=None)


class FullRateDecoder:
    """Opens a clip once and yields every frame in presentation order.

    Attributes:
        timestamps_ns: per-frame presentation time in nanoseconds since the
            first frame, as ``int64``. Not rounded and not ``index / fps``.
        width: frame width in pixels.
        height: frame height in pixels.
        avg_frame_rate: the container's nominal frame rate.
        bit_rate_bps: the input's measured video bitrate, ``0`` if it cannot be
            measured (fewer than two frames).

    """

    def __init__(self, mp4_bytes: bytes) -> None:
        """Index ``mp4_bytes``. No frames are decoded yet."""
        self._sensor = CameraSensor(bytes(mp4_bytes))
        absolute_ns = np.asarray(self._sensor.timestamps_ns, dtype=np.int64)
        self._absolute_ns = absolute_ns
        self.timestamps_ns: npt.NDArray[np.int64] = absolute_ns - absolute_ns[0]
        metadata = self._sensor.video_metadata
        self.width: int = metadata.width
        self.height: int = metadata.height
        self.avg_frame_rate: Fraction = metadata.avg_frame_rate
        self.bit_rate_bps: int = int(metadata.bit_rate_bps)

    @property
    def frame_count(self) -> int:
        """Number of displayable frames."""
        return len(self._absolute_ns)

    def frames(self, window_frames: int | None = None) -> Iterator[npt.NDArray[np.uint8]]:
        """Yield every frame as an ``(H, W, 3)`` RGB ``uint8`` array, in order.

        Args:
            window_frames: decode this many frames per request to bound peak
                memory. ``None`` decodes the whole clip in one request. A smaller
                window re-decodes from the previous keyframe each time, so it
                trades time for memory.

        Raises:
            RuntimeError: if the decoder returns a different number of frames
                than the clip has timestamps. A short decode must not be passed
                on as a clip that was fully checked.

        """
        step = window_frames or self.frame_count
        yielded = 0
        for start in range(0, self.frame_count, step):
            chunk_ns = self._absolute_ns[start : start + step]
            grid = SamplingGrid(
                start_ns=int(chunk_ns[0]),
                exclusive_end_ns=int(chunk_ns[-1]) + 1,
                timestamps_ns=chunk_ns,
                stride_ns=int(chunk_ns[-1]) + 1 - int(chunk_ns[0]),
                duration_ns=int(chunk_ns[-1]) + 1 - int(chunk_ns[0]),
            )
            for batch in self._sensor.sample(SamplingSpec(grid=grid), policy=_NEAREST):
                for frame in batch.frames:
                    yielded += 1
                    yield np.asarray(frame, dtype=np.uint8)
        if yielded != self.frame_count:
            msg = f"decoded {yielded} frames but the clip has {self.frame_count} timestamps"
            raise RuntimeError(msg)
