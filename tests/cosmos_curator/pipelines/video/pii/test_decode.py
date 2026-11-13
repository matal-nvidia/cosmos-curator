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

"""Step 1: the full-rate decode returns every frame with its real timestamp."""

from pathlib import Path

import numpy as np
import pytest

from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.pipelines.video.pii.decode import FullRateDecoder
from cosmos_curator.pipelines.video.tracking.sensor_decode import decode_clip_at_fps
from tests.cosmos_curator.pipelines.video.pii._pii_test_utils import HEIGHT, WIDTH, Scene, encode_test_clip

_SAMPLE_CLIP = Path(__file__).parents[1] / "data" / "test_clip_10s.mp4"
_NS_PER_MS = 1_000_000


def _clip(n_frames: int = 12, timestamps_ms: list[int] | None = None) -> bytes:
    scene = Scene()
    frames = [scene.render((60 + i, 70), camera_shift=(i, 0)) for i in range(n_frames)]
    return encode_test_clip(frames, timestamps_ms)


def test_returns_every_frame() -> None:
    """The frame count equals the clip's frame count, with the right geometry."""
    decoder = FullRateDecoder(_clip(12))
    frames = list(decoder.frames())

    assert len(frames) == decoder.frame_count == 12
    assert all(f.shape == (HEIGHT, WIDTH, 3) and f.dtype == np.uint8 for f in frames)
    assert (decoder.width, decoder.height) == (WIDTH, HEIGHT)


def test_timestamps_are_the_clips_own_not_a_grid() -> None:
    """On a variable-frame-rate clip the timestamps are the real, uneven ones."""
    timestamps_ms = [0, 40, 100, 140, 300, 340, 520, 560]
    decoder = FullRateDecoder(_clip(len(timestamps_ms), timestamps_ms))

    assert decoder.timestamps_ns.dtype == np.int64
    np.testing.assert_array_equal(decoder.timestamps_ns, np.array(timestamps_ms) * _NS_PER_MS)


def test_every_frame_is_kept_where_the_grid_decode_drops_some() -> None:
    """Sampling a VFR clip on a grid loses frames. The full-rate decode does not."""
    timestamps_ms = [0, 40, 100, 140, 300, 340, 520, 560, 900, 940]
    mp4 = _clip(len(timestamps_ms), timestamps_ms)

    grid_frames = decode_clip_at_fps(mp4, target_fps=2.0).frames_rgb
    full_frames = list(FullRateDecoder(mp4).frames())

    assert len(full_frames) == len(timestamps_ms)
    assert len(grid_frames) != len(timestamps_ms)


def test_matches_the_sensors_own_timeline() -> None:
    """Timestamps equal the sensor's presentation timestamps, shifted to start at zero."""
    mp4 = _clip(10)
    sensor_ts = CameraSensor(mp4).timestamps_ns

    decoder = FullRateDecoder(mp4)

    np.testing.assert_array_equal(decoder.timestamps_ns, sensor_ts - sensor_ts[0])


@pytest.mark.parametrize("window_frames", [1, 3, 5, 100])
def test_windowed_decode_gives_the_same_frames(window_frames: int) -> None:
    """A memory-bounding window changes how frames are requested, not which frames come back."""
    mp4 = _clip(12)
    whole = list(FullRateDecoder(mp4).frames())

    windowed = list(FullRateDecoder(mp4).frames(window_frames=window_frames))

    assert len(windowed) == len(whole)
    for a, b in zip(whole, windowed, strict=True):
        np.testing.assert_array_equal(a, b)


def test_frames_arrive_in_presentation_order() -> None:
    """The face moves one pixel per frame, so decoded order is checkable from pixels."""
    scene = Scene()
    frames = [scene.render(None, camera_shift=(i * 2, 0)) for i in range(8)]
    decoder = FullRateDecoder(encode_test_clip(frames))

    decoded = list(decoder.frames())

    # Each decoded frame must be closer to its own source than to any other source.
    for i, decoded_frame in enumerate(decoded):
        errors = [np.abs(decoded_frame.astype(int) - src.astype(int)).mean() for src in frames]
        assert int(np.argmin(errors)) == i


def test_real_sample_clip_has_every_frame() -> None:
    """The repo's 10 s, 24 fps sample decodes all 240 frames at 854x480."""
    decoder = FullRateDecoder(_SAMPLE_CLIP.read_bytes())

    assert decoder.frame_count == 240
    assert (decoder.width, decoder.height) == (854, 480)
    assert sum(1 for _ in decoder.frames(window_frames=60)) == 240
