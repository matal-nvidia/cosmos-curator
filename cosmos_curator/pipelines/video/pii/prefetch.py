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

"""Run the detector on its own thread, ahead of the tracker.

The tracker asks for a look and waits for the answer. Here the look is started
as soon as its frame is decoded, a few frames before the tracker reaches it, so
the GPU works while the calling thread tracks, blurs and encodes. One worker
thread owns the model, so looks still run one at a time and in frame order.
"""

import collections
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import numpy.typing as npt

from cosmos_curator.pipelines.video.pii.detector import Detection, PiiDetector


class PrefetchingDetector:
    """A :class:`PiiDetector` that answers looks started early by :meth:`frames`."""

    def __init__(self, detector: PiiDetector, look_interval: int, lookahead_frames: int) -> None:
        """Wrap ``detector``. Looks are frames ``0, look_interval, 2 * look_interval, ...``."""
        if lookahead_frames < 1:
            msg = f"lookahead_frames must be >= 1, got {lookahead_frames}"
            raise ValueError(msg)
        self._detector = detector
        self._look_interval = look_interval
        self._lookahead = lookahead_frames
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pii-detector")
        self._pending: dict[int, Future[list[Detection]]] = {}

    def frames(self, frames: Iterable[npt.NDArray[np.uint8]]) -> Iterator[npt.NDArray[np.uint8]]:
        """Yield ``frames`` in order, starting each look ``lookahead_frames`` before its frame is yielded."""
        window: collections.deque[npt.NDArray[np.uint8]] = collections.deque()
        for index, frame in enumerate(frames):
            if index % self._look_interval == 0:
                self._pending[index] = self._pool.submit(self._detector.detect, frame, index)
            window.append(frame)
            if len(window) > self._lookahead:
                yield window.popleft()
        while window:
            yield window.popleft()

    def detect(self, frame_rgb: npt.NDArray[np.uint8], frame_index: int) -> list[Detection]:
        """Return the look started for ``frame_index``, or run one now on the worker."""
        future = self._pending.pop(frame_index, None)
        if future is None:
            future = self._pool.submit(self._detector.detect, frame_rgb, frame_index)
        return future.result()

    def close(self) -> None:
        """Drop looks nobody asked for and wait for the worker to stop."""
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()
        self._pool.shutdown(wait=True)
