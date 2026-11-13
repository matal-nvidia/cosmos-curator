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

"""Synthetic scenes for the PII tests: noise background, noise "face", known masks.

Lucas-Kanade needs texture, so both the background and the face are random
noise. The face is a textured disc whose centre is chosen per frame, so the
ground-truth mask is exact and the fake detector can return it directly.
"""

import io
from collections.abc import Callable, Sequence
from fractions import Fraction

import av
import numpy as np
import numpy.typing as npt

from cosmos_curator.pipelines.video.pii.detector import Detection

WIDTH = 320
HEIGHT = 240
FACE_RADIUS = 40  # an 80 px face, well over the 21 px Lucas-Kanade window

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive end)
Center = tuple[float, float]


def disc_mask(
    center: Center, radius: int = FACE_RADIUS, shape: tuple[int, int] = (HEIGHT, WIDTH)
) -> npt.NDArray[np.bool_]:
    """Boolean disc of ``radius`` around the integer-rounded ``center`` (x, y)."""
    cx, cy = round(center[0]), round(center[1])
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2


def box_mask(box: Box, shape: tuple[int, int] = (HEIGHT, WIDTH)) -> npt.NDArray[np.bool_]:
    """Boolean rectangle."""
    mask = np.zeros(shape, dtype=bool)
    mask[box[1] : box[3], box[0] : box[2]] = True
    return mask


class Scene:
    """A noise background with an optional noise face at a chosen centre."""

    def __init__(self, seed: int = 0) -> None:
        """Draw the background and face textures once so every frame shares them."""
        rng = np.random.default_rng(seed)
        self._background = rng.integers(0, 256, size=(HEIGHT * 2, WIDTH * 2, 3), dtype=np.uint8)
        # The tile travels with the face, so the face has motion of its own to follow.
        self._face_tile = rng.integers(0, 256, size=(2 * FACE_RADIUS + 1, 2 * FACE_RADIUS + 1, 3), dtype=np.uint8)

    def render(self, face_center: Center | None, camera_shift: tuple[int, int] = (0, 0)) -> npt.NDArray[np.uint8]:
        """Render one frame. The camera shifts the background and the face together."""
        dx, dy = camera_shift
        frame = self._background[50 + dy : 50 + dy + HEIGHT, 50 + dx : 50 + dx + WIDTH].copy()
        if face_center is not None:
            cx, cy = round(face_center[0]), round(face_center[1])
            ys, xs = np.nonzero(disc_mask(face_center))
            frame[ys, xs] = self._face_tile[ys - cy + FACE_RADIUS, xs - cx + FACE_RADIUS]
        return frame


class ScriptedDetector:
    """Returns the masks we scripted for the frame it is asked about, and records every call.

    ``script`` maps a frame index to the detections returned on that frame; any
    frame not in the script returns no detections.
    """

    def __init__(self, script: dict[int, list[Detection]] | Callable[[int], list[Detection]]) -> None:
        """Store the script and start with no recorded calls."""
        self._script = script
        self.calls: list[int] = []

    def detect(self, frame_rgb: npt.NDArray[np.uint8], frame_index: int) -> list[Detection]:  # noqa: ARG002
        """Record the call and return the scripted detections."""
        self.calls.append(frame_index)
        if callable(self._script):
            return self._script(frame_index)
        return list(self._script.get(frame_index, []))


def face(mask: npt.NDArray[np.bool_], score: float = 0.9, label: str = "human face") -> Detection:
    """Build a detection with a chosen mask and score."""
    return Detection(mask=mask, score=score, label=label)


def encode_test_clip(
    frames: Sequence[npt.NDArray[np.uint8]],
    timestamps_ms: Sequence[int] | None = None,
    *,
    fps: int = 24,
    bit_rate: int = 2_000_000,
) -> bytes:
    """Encode ``frames`` as an H.264 MP4, optionally at uneven (VFR) timestamps."""
    if timestamps_ms is None:
        timestamps_ms = [round(i * 1000 / fps) for i in range(len(frames))]
    time_base = Fraction(1, 90000)
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libopenh264", rate=Fraction(fps, 1))
        stream.width = frames[0].shape[1]
        stream.height = frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = bit_rate
        stream.codec_context.time_base = time_base
        for frame, ts_ms in zip(frames, timestamps_ms, strict=True):
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = round(ts_ms * 90)
            video_frame.time_base = time_base
            container.mux(stream.encode(video_frame))
        container.mux(stream.encode(None))
    return buffer.getvalue()


def iou(a: npt.NDArray[np.bool_], b: npt.NDArray[np.bool_]) -> float:
    """Intersection over union of two boolean masks."""
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else int(np.count_nonzero(a & b)) / union
