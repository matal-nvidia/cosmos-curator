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

"""Follow a mask from the frame where it was detected to another frame.

Every warp starts from the mask at the last real detection (the anchor) and goes
straight to the target frame. A warped mask is never warped again, so error does
not stack up frame over frame. The same call serves forward frames (between
looks) and backward frames (painting a newly found face back onto held frames).

The transform is a rigid-plus-scale fit (rotation, uniform scale, translation)
to sparse Lucas-Kanade flow of corners found inside the anchor mask.
"""

from typing import Protocol

import attrs
import cv2
import numpy as np
import numpy.typing as npt

from cosmos_curator.pipelines.video.pii.config import LKConfig


@attrs.define(frozen=True, eq=False)
class Anchor:
    """The image and corners a mask is followed from.

    Attributes:
        gray: the anchor frame in grayscale.
        points: ``(N, 1, 2)`` float32 corners inside the anchor mask; may be empty.

    """

    gray: npt.NDArray[np.uint8]
    points: npt.NDArray[np.float32]


class MaskPropagator(Protocol):
    """Finds the 2x3 transform that carries an anchor's mask onto another frame."""

    def prepare(self, gray: npt.NDArray[np.uint8], mask: npt.NDArray[np.bool_]) -> Anchor:
        """Search for corners in ``mask`` on ``gray`` once, when a face is detected."""
        ...

    def estimate(self, anchor: Anchor, target_gray: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64] | None:
        """Return a ``(2, 3)`` transform anchor -> target, or ``None`` if it cannot be trusted."""
        ...


class LKRigidPropagator:
    """Sparse Lucas-Kanade flow plus a RANSAC rigid-and-scale fit."""

    def __init__(self, config: LKConfig | None = None) -> None:
        """Use ``config`` or the prototype defaults."""
        self._config = config or LKConfig()

    def prepare(self, gray: npt.NDArray[np.uint8], mask: npt.NDArray[np.bool_]) -> Anchor:
        """Find corners inside ``mask``."""
        cfg = self._config
        # OpenCV returns None when it finds no corners; its stubs say it never does.
        found: object = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=cfg.max_features,
            qualityLevel=cfg.feature_quality,
            minDistance=cfg.feature_min_distance,
            mask=mask.astype(np.uint8),
        )
        points = np.empty((0, 1, 2), dtype=np.float32) if found is None else np.asarray(found, dtype=np.float32)
        return Anchor(gray=gray, points=points)

    def estimate(self, anchor: Anchor, target_gray: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64] | None:
        """Return the transform, or ``None`` on too few corners, lost flow, or a failed fit."""
        cfg = self._config
        if len(anchor.points) < cfg.min_features:
            return None
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            anchor.gray,
            target_gray,
            anchor.points,
            np.empty_like(anchor.points),
            winSize=(cfg.window_size, cfg.window_size),
            maxLevel=cfg.pyramid_levels,
        )
        tracked = status.ravel() == 1
        if int(tracked.sum()) < cfg.min_features:
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(
            anchor.points[tracked].reshape(-1, 2),
            moved[tracked].reshape(-1, 2),
            method=cv2.RANSAC,
        )
        if matrix is None or inliers is None or int(inliers.sum()) < cfg.min_features:
            return None
        return np.asarray(matrix, dtype=np.float64)


def warp_mask(mask: npt.NDArray[np.bool_], matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Carry ``mask`` through the ``(2, 3)`` ``matrix``; pixels moved off the frame are lost."""
    height, width = mask.shape
    warped = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped.astype(bool)
