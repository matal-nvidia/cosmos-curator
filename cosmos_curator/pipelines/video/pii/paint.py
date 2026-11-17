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

"""Turn the tracker's masks into pixels.

Every paint mode grows the mask the same way and differs only in what is written
into the frame. The growth factor, the kernel and the mode come from
:class:`~cosmos_curator.pipelines.video.pii.config.PaintConfig`, which has no
defaults for the growth factors.
"""

from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt

from cosmos_curator.pipelines.video.pii.config import PaintConfig, PaintMode
from cosmos_curator.pipelines.video.pii.tracker import Region


def grow_mask(mask: npt.NDArray[np.bool_], scale: float) -> npt.NDArray[np.bool_]:
    """Scale ``mask`` about its centroid by ``scale``. The result always contains ``mask``.

    ``1.0`` returns the mask unchanged and ``1.1`` makes it 10% larger in each
    direction. The original is OR-ed back in, so a shape that is not centred on
    its centroid can never lose coverage by being grown.
    """
    if scale == 1.0:
        return mask
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    cx, cy = float(xs.mean()), float(ys.mean())
    matrix = np.array([[scale, 0.0, cx * (1.0 - scale)], [0.0, scale, cy * (1.0 - scale)]], dtype=np.float64)
    height, width = mask.shape
    grown = cv2.warpAffine(
        mask.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    ).astype(bool)
    return grown | mask


def region_scale(region: Region, config: PaintConfig) -> float:
    """Return the total growth for ``region``: the base growth, plus one more step per missed look."""
    return config.mask_growth * config.missed_look_growth**region.missed_looks


def union_mask(regions: Sequence[Region], config: PaintConfig) -> npt.NDArray[np.bool_] | None:
    """Return the grown masks of every region on a frame OR-ed together; ``None`` if there are none."""
    if not regions:
        return None
    union = grow_mask(regions[0].mask, region_scale(regions[0], config))
    for region in regions[1:]:
        union = union | grow_mask(region.mask, region_scale(region, config))
    return union


# Modes whose filter needs a neighbourhood, so the crop is padded by half the kernel.
_KERNEL_MODES = frozenset({PaintMode.GAUSSIAN, PaintMode.STACK, PaintMode.BOX})


def paint_frame(
    frame_rgb: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_] | None,
    config: PaintConfig,
) -> npt.NDArray[np.uint8]:
    """Apply the paint mode to ``frame_rgb`` under ``mask``.

    ``none`` and an empty mask return the frame itself. Every other mode returns a
    new frame, so the caller's decoded pixels are not edited in place.
    """
    if mask is None or config.mode is PaintMode.NONE:
        return frame_rgb
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return frame_rgb
    kernel = config.blur_kernel if config.mode in _KERNEL_MODES else None
    height, width = mask.shape
    pad = 0 if kernel is None else kernel // 2
    # Filter only a box around the mask. For a kernel mode the box reaches `pad`
    # pixels past it, so every masked pixel sees all its neighbours and the result
    # equals filtering the whole frame (OpenCV's border handling then only ever
    # applies at the real image edge).
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + 1 + pad, height)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + 1 + pad, width)
    redacted = _redact_crop(np.ascontiguousarray(frame_rgb[y0:y1, x0:x1]), config)
    out = frame_rgb.copy()
    box = out[y0:y1, x0:x1]
    inside = mask[y0:y1, x0:x1]
    box[inside] = redacted[inside]
    return out


def _redact_crop(crop: npt.NDArray[np.uint8], config: PaintConfig) -> npt.NDArray[np.uint8]:
    """Cover the crop. The caller copies the result back under the mask only."""
    mode = config.mode
    if mode is PaintMode.SOLID:
        return np.zeros_like(crop)
    if mode is PaintMode.PIXELATE:
        block = config.pixelate_block
        assert block is not None  # PaintConfig guarantees this for pixelate
        return _pixelate(crop, block)
    kernel = config.blur_kernel
    assert kernel is not None  # PaintConfig guarantees this for the kernel modes
    if mode is PaintMode.STACK:
        return np.asarray(cv2.stackBlur(crop, (kernel, kernel)), dtype=np.uint8)
    if mode is PaintMode.BOX:
        return np.asarray(cv2.blur(crop, (kernel, kernel)), dtype=np.uint8)
    return _gaussian(crop, kernel, config.blur_downscale)


def _gaussian(crop: npt.NDArray[np.uint8], kernel: int, downscale: int) -> npt.NDArray[np.uint8]:
    if downscale == 1:
        return np.asarray(cv2.GaussianBlur(crop, (kernel, kernel), 0), dtype=np.uint8)
    height, width = crop.shape[:2]
    small = cv2.resize(crop, (max(1, width // downscale), max(1, height // downscale)), interpolation=cv2.INTER_AREA)
    small_kernel = (kernel // downscale) | 1
    small = cv2.GaussianBlur(small, (small_kernel, small_kernel), 0)
    return np.asarray(cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR), dtype=np.uint8)


def _pixelate(crop: npt.NDArray[np.uint8], block: int) -> npt.NDArray[np.uint8]:
    height, width = crop.shape[:2]
    small = cv2.resize(crop, (max(1, width // block), max(1, height // block)), interpolation=cv2.INTER_AREA)
    return np.asarray(cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST), dtype=np.uint8)
