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

"""The detector thread: looks start early, come back in order, and failures still reach the caller."""

import threading

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.pipelines.video.pii.detector import Detection
from cosmos_curator.pipelines.video.pii.prefetch import PrefetchingDetector


def _frames(n: int) -> list[npt.NDArray[np.uint8]]:
    return [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(n)]


class _Recorder:
    def __init__(self) -> None:
        self.started: list[int] = []
        self.lock = threading.Lock()

    def detect(self, frame_rgb: npt.NDArray[np.uint8], frame_index: int) -> list[Detection]:
        with self.lock:
            self.started.append(frame_index)
        assert int(frame_rgb[0, 0, 0]) == frame_index
        return [Detection(mask=np.zeros((4, 4), dtype=bool), score=1.0)]


def test_a_look_is_started_before_its_frame_reaches_the_tracker() -> None:
    """When frame 0 comes out, the looks on frames up to the lookahead are already queued."""
    inner = _Recorder()
    prefetch = PrefetchingDetector(inner, look_interval=5, lookahead_frames=10)
    try:
        frames = prefetch.frames(_frames(30))
        next(frames)
        assert sorted(prefetch._pending) == [0, 5, 10]
        rest = list(frames)
        assert len(rest) == 29
    finally:
        prefetch.close()


def test_every_frame_comes_out_once_in_order() -> None:
    """The lookahead buffers frames. It must not drop, repeat or reorder one."""
    prefetch = PrefetchingDetector(_Recorder(), look_interval=5, lookahead_frames=7)
    try:
        out = [int(frame[0, 0, 0]) for frame in prefetch.frames(_frames(23))]
    finally:
        prefetch.close()

    assert out == list(range(23))


def test_detect_returns_the_look_for_the_frame_asked() -> None:
    """The answer for frame 10 is the look on frame 10, and an unplanned frame still gets a look."""
    inner = _Recorder()
    prefetch = PrefetchingDetector(inner, look_interval=5, lookahead_frames=10)
    try:
        frames = list(prefetch.frames(_frames(12)))
        assert prefetch.detect(frames[10], 10)
        assert prefetch.detect(frames[11], 11)
    finally:
        prefetch.close()

    assert 11 in inner.started
    assert inner.started.count(10) == 1


def test_a_failed_look_is_raised_to_whoever_asks_for_it() -> None:
    """A model error on the worker comes back as the same error, not as an empty look."""

    class Broken:
        def detect(self, frame_rgb: npt.NDArray[np.uint8], frame_index: int) -> list[Detection]:  # noqa: ARG002
            msg = "model fell over"
            raise RuntimeError(msg)

    prefetch = PrefetchingDetector(Broken(), look_interval=5, lookahead_frames=10)
    try:
        frames = list(prefetch.frames(_frames(6)))
        with pytest.raises(RuntimeError, match="model fell over"):
            prefetch.detect(frames[0], 0)
    finally:
        prefetch.close()


def test_a_lookahead_under_one_is_refused() -> None:
    """0 is handled by not using this class at all."""
    with pytest.raises(ValueError, match="lookahead_frames"):
        PrefetchingDetector(_Recorder(), look_interval=5, lookahead_frames=0)
