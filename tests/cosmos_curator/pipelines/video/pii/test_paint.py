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

"""Step 3: the paint switch and mask growth.

Growth and kernel size are arguments with no defaults, because the plan has not
chosen them. The tests pass arbitrary values; none of them is a recommendation.
"""

from collections.abc import Callable

import cv2
import numpy as np
import pytest

from cosmos_curator.pipelines.video.pii.config import PaintConfig, PaintMode
from cosmos_curator.pipelines.video.pii.paint import grow_mask, paint_frame, region_scale, union_mask
from cosmos_curator.pipelines.video.pii.tracker import Region
from tests.cosmos_curator.pipelines.video.pii._pii_test_utils import HEIGHT, WIDTH, Scene, box_mask, disc_mask

CENTER = (160, 120)
FACE = disc_mask(CENTER)

# Arbitrary values used only to exercise the code. They are not defaults and not advice.
_GROWTH = 1.25
_MISSED = 1.5
_KERNEL = 31


def _gaussian(growth: float = 1.0, missed: float = 1.0, kernel: int = _KERNEL) -> PaintConfig:
    return PaintConfig(mode=PaintMode.GAUSSIAN, mask_growth=growth, missed_look_growth=missed, blur_kernel=kernel)


def _none(growth: float = 1.0, missed: float = 1.0) -> PaintConfig:
    return PaintConfig(mode=PaintMode.NONE, mask_growth=growth, missed_look_growth=missed)


def _region(mask: np.ndarray, missed: int = 0, track_id: int = 0) -> Region:
    return Region(track_id=track_id, label="human face", mask=mask, missed_looks=missed)


# -------------------------------------------------------------------- config


def test_growth_has_no_default() -> None:
    """The growth factors are required: leaving them out is an error, not a silent choice."""
    with pytest.raises(TypeError):
        PaintConfig(mode=PaintMode.NONE)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        PaintConfig(mode=PaintMode.NONE, mask_growth=1.0)  # type: ignore[call-arg]


def test_gaussian_requires_a_kernel_and_has_no_default() -> None:
    """Gaussian without a kernel size is rejected."""
    with pytest.raises(ValueError, match="blur_kernel is required"):
        PaintConfig(mode=PaintMode.GAUSSIAN, mask_growth=1.0, missed_look_growth=1.0)


@pytest.mark.parametrize("mode", [PaintMode.GAUSSIAN, PaintMode.STACK, PaintMode.BOX])
@pytest.mark.parametrize("kernel", [0, -3, 4, 30])
def test_kernel_must_be_odd_and_positive(mode: PaintMode, kernel: int) -> None:
    """Even and non-positive kernels are rejected for every blur that uses one."""
    with pytest.raises(ValueError, match="odd"):
        PaintConfig(mode=mode, mask_growth=1.0, missed_look_growth=1.0, blur_kernel=kernel)


@pytest.mark.parametrize("mode", [PaintMode.STACK, PaintMode.BOX])
def test_stack_and_box_require_a_kernel(mode: PaintMode) -> None:
    """The other kernel blurs are refused the same way gaussian is."""
    with pytest.raises(ValueError, match="blur_kernel is required"):
        PaintConfig(mode=mode, mask_growth=1.0, missed_look_growth=1.0)


def test_pixelate_requires_a_block() -> None:
    """Pixelate with no block size is rejected."""
    with pytest.raises(ValueError, match="pixelate_block"):
        PaintConfig(mode=PaintMode.PIXELATE, mask_growth=1.0, missed_look_growth=1.0)


@pytest.mark.parametrize(("downscale", "kernel"), [(0, 31), (33, 31)])
def test_downscale_must_be_between_one_and_the_kernel(downscale: int, kernel: int) -> None:
    """A factor under 1 is meaningless, and one past the kernel leaves nothing to blur with."""
    with pytest.raises(ValueError, match="blur_downscale"):
        PaintConfig(
            mode=PaintMode.GAUSSIAN,
            mask_growth=1.0,
            missed_look_growth=1.0,
            blur_kernel=kernel,
            blur_downscale=downscale,
        )


def test_none_mode_needs_no_kernel() -> None:
    """The kernel only matters when something is blurred."""
    assert _none().blur_kernel is None


@pytest.mark.parametrize(("growth", "missed"), [(0.9, 1.0), (1.0, 0.5)])
def test_growth_below_one_is_rejected(growth: float, missed: float) -> None:
    """A mask never shrinks: 1.0 means no growth."""
    with pytest.raises(ValueError, match=r">= 1\.0"):
        PaintConfig(mode=PaintMode.NONE, mask_growth=growth, missed_look_growth=missed)


# -------------------------------------------------------------------- growth


def test_growth_of_one_changes_nothing() -> None:
    """1.0 means no growth, the way EgoBlur's scale factor does."""
    assert np.array_equal(grow_mask(FACE, 1.0), FACE)


def test_grown_mask_contains_the_original_and_is_larger() -> None:
    """Growing never loses a pixel and the area goes up by about the scale squared."""
    grown = grow_mask(FACE, 1.5)

    assert np.all(grown[FACE])
    ratio = np.count_nonzero(grown) / np.count_nonzero(FACE)
    assert ratio == pytest.approx(1.5**2, rel=0.05)


def test_growth_is_about_the_centroid() -> None:
    """The grown disc is still centred on the same point."""
    grown = grow_mask(FACE, 1.5)
    ys, xs = np.nonzero(grown)

    assert xs.mean() == pytest.approx(CENTER[0], abs=0.5)
    assert ys.mean() == pytest.approx(CENTER[1], abs=0.5)


def test_growth_never_uncovers_a_lopsided_shape() -> None:
    """An L shape is not star-shaped about its centroid; the original is OR-ed back so nothing is lost."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[50:150, 50:70] = True
    mask[130:150, 50:150] = True

    grown = grow_mask(mask, 1.2)

    assert np.all(grown[mask])


def test_growing_an_empty_mask_is_harmless() -> None:
    """No pixels in, no pixels out."""
    assert not grow_mask(np.zeros((HEIGHT, WIDTH), dtype=bool), 2.0).any()


def test_missed_looks_grow_the_mask_more() -> None:
    """Each missed look multiplies the growth, because the position is less sure."""
    cfg = _none(growth=_GROWTH, missed=_MISSED)

    assert region_scale(_region(FACE, missed=0), cfg) == pytest.approx(_GROWTH)
    assert region_scale(_region(FACE, missed=1), cfg) == pytest.approx(_GROWTH * _MISSED)
    assert region_scale(_region(FACE, missed=2), cfg) == pytest.approx(_GROWTH * _MISSED**2)


def test_union_covers_every_region() -> None:
    """Two faces on a frame give one mask that covers both."""
    left, right = box_mask((20, 20, 80, 80)), box_mask((200, 20, 260, 80))

    union = union_mask([_region(left, track_id=0), _region(right, track_id=1)], _none())

    assert union is not None
    assert np.array_equal(union, left | right)


def test_union_of_no_regions_is_none() -> None:
    """A frame with no faces has no mask."""
    assert union_mask([], _none()) is None


# --------------------------------------------------------------------- paint


def _frame() -> np.ndarray:
    return Scene().render(CENTER)


def test_none_leaves_every_pixel_alone() -> None:
    """The no-blur mode returns the frame untouched."""
    frame = _frame()

    out = paint_frame(frame, FACE, _none())

    assert out is frame


def test_gaussian_changes_only_pixels_under_the_mask() -> None:
    """Outside the mask every pixel is bit-for-bit the original."""
    frame = _frame()
    before = frame.copy()

    out = paint_frame(frame, FACE, _gaussian())

    assert np.array_equal(out[~FACE], before[~FACE])
    assert not np.array_equal(out[FACE], before[FACE])


def test_gaussian_does_not_edit_the_input_in_place() -> None:
    """The decoded frame stays as decoded; a blurred copy comes back."""
    frame = _frame()
    before = frame.copy()

    out = paint_frame(frame, FACE, _gaussian())

    assert np.array_equal(frame, before)
    assert out is not frame


def test_gaussian_removes_the_detail_inside_the_mask() -> None:
    """The face is noise, so its spread collapses once it is blurred."""
    frame = _frame()

    out = paint_frame(frame, FACE, _gaussian())

    assert out[FACE].std() < 0.2 * frame[FACE].std()


@pytest.mark.parametrize(
    "mask",
    [
        FACE,
        box_mask((0, 0, 60, 60)),
        box_mask((WIDTH - 60, HEIGHT - 60, WIDTH, HEIGHT)),
        box_mask((0, 100, WIDTH, 140)),
    ],
    ids=["centre", "top-left-corner", "bottom-right-corner", "full-width-band"],
)
def test_cropped_blur_equals_blurring_the_whole_frame(mask: np.ndarray) -> None:
    """Blurring a box around the mask is an optimisation, so it must not change a single pixel."""
    frame = _frame()
    reference = frame.copy()
    reference[mask] = cv2.GaussianBlur(frame, (_KERNEL, _KERNEL), 0)[mask]

    out = paint_frame(frame, mask, _gaussian())

    assert np.array_equal(out, reference)


def _downscaled(kernel: int = _KERNEL, downscale: int = 4) -> PaintConfig:
    return PaintConfig(
        mode=PaintMode.GAUSSIAN,
        mask_growth=1.0,
        missed_look_growth=1.0,
        blur_kernel=kernel,
        blur_downscale=downscale,
    )


def test_downscaled_blur_changes_only_pixels_under_the_mask() -> None:
    """The cheaper blur keeps the same boundary: outside the mask nothing moves."""
    frame = _frame()
    before = frame.copy()

    out = paint_frame(frame, FACE, _downscaled())

    assert np.array_equal(frame, before)
    assert np.array_equal(out[~FACE], before[~FACE])


def test_downscaled_blur_removes_the_detail_inside_the_mask() -> None:
    """Shrinking first must not leave the face readable."""
    frame = _frame()

    out = paint_frame(frame, FACE, _downscaled())

    assert out[FACE].std() < 0.2 * frame[FACE].std()


def test_downscaled_blur_is_close_to_the_exact_one() -> None:
    """Close, not equal: the average pixel under the mask moves by a few levels at most."""
    frame = _frame()

    exact = paint_frame(frame, FACE, _gaussian(kernel=151))
    cheap = paint_frame(frame, FACE, _downscaled(kernel=151, downscale=4))

    assert np.abs(cheap[FACE].astype(np.int16) - exact[FACE].astype(np.int16)).mean() < 3.0


def _mode(mode: PaintMode) -> PaintConfig:
    return PaintConfig(
        mode=mode,
        mask_growth=1.0,
        missed_look_growth=1.0,
        blur_kernel=_KERNEL,
        pixelate_block=8,
    )


@pytest.mark.parametrize("mode", [PaintMode.STACK, PaintMode.BOX, PaintMode.PIXELATE, PaintMode.SOLID])
def test_each_redaction_changes_only_pixels_under_the_mask(mode: PaintMode) -> None:
    """Every style keeps the same boundary: outside the mask nothing moves."""
    frame = _frame()
    before = frame.copy()

    out = paint_frame(frame, FACE, _mode(mode))

    assert np.array_equal(frame, before)
    assert np.array_equal(out[~FACE], before[~FACE])
    assert not np.array_equal(out[FACE], before[FACE])


def test_solid_paints_the_mask_black() -> None:
    """Solid fill is black under the mask and nowhere else."""
    frame = _frame()

    out = paint_frame(frame, FACE, _mode(PaintMode.SOLID))

    assert int(out[FACE].max()) == 0


@pytest.mark.parametrize(
    ("mode", "filter_frame"),
    [
        (PaintMode.STACK, lambda frame: cv2.stackBlur(frame, (_KERNEL, _KERNEL))),
        (PaintMode.BOX, lambda frame: cv2.blur(frame, (_KERNEL, _KERNEL))),
    ],
)
def test_cropped_kernel_blur_equals_filtering_the_whole_frame(
    mode: PaintMode, filter_frame: Callable[[np.ndarray], np.ndarray]
) -> None:
    """The padded crop is an optimisation, so stack and box must match a full-frame filter."""
    frame = _frame()
    reference = frame.copy()
    reference[FACE] = filter_frame(frame)[FACE]

    out = paint_frame(frame, FACE, _mode(mode))

    assert np.array_equal(out, reference)


def test_empty_mask_paints_nothing() -> None:
    """No mask, or an empty one, returns the frame."""
    frame = _frame()

    assert paint_frame(frame, None, _gaussian()) is frame
    assert paint_frame(frame, np.zeros((HEIGHT, WIDTH), dtype=bool), _gaussian()) is frame
