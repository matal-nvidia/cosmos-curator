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

"""Step 2: the tracker, tested against a fake detector that returns the masks we chose.

These tests are the design. Each one states a rule from the plan and checks the
tracker against it. The detector is scripted, so a wrong mask can only be the
tracker's fault.
"""

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.pipelines.video.pii.config import LKConfig, TrackerConfig
from cosmos_curator.pipelines.video.pii.propagate import Anchor, LKRigidPropagator
from cosmos_curator.pipelines.video.pii.tracker import FrameResult, PiiTracker
from tests.cosmos_curator.pipelines.video.pii._pii_test_utils import (
    Scene,
    ScriptedDetector,
    box_mask,
    disc_mask,
    face,
    iou,
)

CENTER = (160, 120)
FACE = disc_mask(CENTER)


def _static_frames(n: int, center: tuple[int, int] | None = CENTER) -> list[npt.NDArray[np.uint8]]:
    scene = Scene()
    return [scene.render(center) for _ in range(n)]


def _run(
    detector: ScriptedDetector,
    frames: list[npt.NDArray[np.uint8]],
    config: TrackerConfig | None = None,
    propagator: object | None = None,
) -> tuple[list[FrameResult], PiiTracker, dict[int, int]]:
    """Push every frame. Returns all results, the tracker, and ``{frame index: push that released it}``."""
    tracker = PiiTracker(detector, config, propagator)  # type: ignore[arg-type]
    results: list[FrameResult] = []
    released_at: dict[int, int] = {}
    for push_number, frame in enumerate(frames):
        for result in tracker.push(frame):
            released_at[result.index] = push_number
            results.append(result)
    for result in tracker.finish():
        released_at[result.index] = len(frames)
        results.append(result)
    return results, tracker, released_at


def _regions(results: list[FrameResult], index: int) -> list:
    return next(r for r in results if r.index == index).regions


def _painted(results: list[FrameResult]) -> list[int]:
    return [r.index for r in results if r.regions]


# --------------------------------------------------------------------- looks


def test_detector_looks_on_frames_0_5_10() -> None:
    """The detector runs on every 5th frame, and on the last frame when that is not already a look."""
    detector = ScriptedDetector({})

    _run(detector, _static_frames(18, center=None))

    assert detector.calls == [0, 5, 10, 15, 17]


def test_every_frame_comes_back_once_in_order() -> None:
    """No frame is lost, repeated, or reordered on the way through the hold."""
    results, tracker, _ = _run(ScriptedDetector({}), _static_frames(23, center=None))

    assert [r.index for r in results] == list(range(23))
    assert tracker.stats.frames == 23


def test_no_face_means_no_regions() -> None:
    """With nothing detected, nothing is masked."""
    results, _, _ = _run(ScriptedDetector({}), _static_frames(20, center=None))

    assert _painted(results) == []


# ----------------------------------------------------------------- seen twice


def test_face_seen_once_is_never_blurred() -> None:
    """A face found on one look and not on the next is not blurred anywhere."""
    detector = ScriptedDetector({15: [face(FACE)]})  # look 3 only

    results, tracker, _ = _run(detector, _static_frames(40))

    assert _painted(results) == []
    assert tracker.stats.tracks_created == 1
    assert tracker.stats.tracks_confirmed == 0
    assert tracker.stats.tracks_dropped_unconfirmed == 1


def test_face_seen_on_two_looks_in_a_row_is_blurred() -> None:
    """The second look confirms the face, and the frames that were held get its mask."""
    detector = ScriptedDetector({15: [face(FACE)], 20: [face(FACE)], 25: [face(FACE)]})

    results, tracker, _ = _run(detector, _static_frames(40))

    assert tracker.stats.tracks_confirmed == 1
    # First seen at 15, so the held frames before it are painted too, not only 15 onward.
    assert 15 in _painted(results)
    assert 14 in _painted(results)


def test_first_two_looks_are_exempt_from_the_wait() -> None:
    """A face found only on look 0 or only on look 1 is blurred at once."""
    for only_look_frame in (0, 5):
        detector = ScriptedDetector({only_look_frame: [face(FACE)]})

        results, tracker, _ = _run(detector, _static_frames(30))

        assert tracker.stats.tracks_confirmed == 1, only_look_frame
        assert only_look_frame in _painted(results)


def test_the_last_frame_is_a_look_when_it_is_not_already_one() -> None:
    """A face that enters after the final scheduled look is detected, blurred, and painted backward."""
    detector = ScriptedDetector({7: [face(FACE)]})

    results, tracker, _ = _run(detector, _static_frames(8))

    assert detector.calls == [0, 5, 7]
    assert tracker.stats.tracks_confirmed == 1
    # The clip is shorter than the hold, so the closing look paints the whole clip.
    assert _painted(results) == list(range(8))


def test_the_last_frame_is_not_looked_at_twice() -> None:
    """A clip that already ends on a look does not detect that frame again."""
    detector = ScriptedDetector({})

    _run(detector, _static_frames(16, center=None))

    assert detector.calls == [0, 5, 10, 15]


def test_look_two_is_not_exempt() -> None:
    """The exemption stops after the first two looks: a face found only on look 2 is not blurred."""
    detector = ScriptedDetector({10: [face(FACE)]})

    results, tracker, _ = _run(detector, _static_frames(40))

    assert tracker.stats.tracks_confirmed == 0
    assert _painted(results) == []


# ------------------------------------------------------------- paint backward


def test_confirmed_face_is_painted_back_onto_the_frames_it_was_already_in() -> None:
    """A face that walks in mid-gap is found on a later look and painted back onto the earlier frames."""
    scene = Scene()
    entered_at = 12  # in the picture from frame 12; the look on frame 10 sees nothing
    frames = [scene.render((CENTER[0] + i, CENTER[1]) if i >= entered_at else None) for i in range(40)]

    def truth(i: int) -> npt.NDArray[np.bool_]:
        return disc_mask((CENTER[0] + i, CENTER[1]))

    detector = ScriptedDetector({15: [face(truth(15))], 20: [face(truth(20))]})
    results, _, _ = _run(detector, frames)

    # Frames 12..14 are before the first look that saw the face (15).
    for i in (12, 13, 14):
        regions = _regions(results, i)
        assert len(regions) == 1, i
        assert iou(regions[0].mask, truth(i)) > 0.8, i


def test_painting_back_reaches_as_far_as_the_hold_allows_after_the_second_look() -> None:
    """The face is in the picture the whole time, but the early looks missed it.

    First sighting is frame 20 and the second look (the confirmation) is frame 25.
    A frame is released when frame j + 16 is pushed, so at frame 25 the oldest frame
    that can still change is frame 9. Frames 0..8 left the hold while the face was
    still unconfirmed, so they stay clear. The wait for the second look uses up
    5 of the 15 held frames.
    """
    detector = ScriptedDetector({20: [face(FACE)], 25: [face(FACE)]})

    results, _, _ = _run(detector, _static_frames(40))

    painted = _painted(results)
    assert painted[0] == 9
    assert all(i in painted for i in range(9, 30))
    assert not any(i in painted for i in range(9))


# ---------------------------------------------------------- blur on leaving hold


def test_frame_is_released_only_when_it_leaves_the_hold() -> None:
    """Frame j is released exactly when frame j + hold + 1 is pushed, never earlier, never later."""
    hold = TrackerConfig().hold_frames
    frames = _static_frames(40, center=None)

    _, _, released_at = _run(ScriptedDetector({}), frames)

    for index in range(40 - hold - 1):
        assert released_at[index] == index + hold + 1
    # The last frames are only released when the clip ends.
    assert all(released_at[i] == 40 for i in range(40 - hold - 1, 40))


def test_a_released_frame_is_never_changed_later() -> None:
    """Masks on a frame as released are the masks the tracker would have given at the end."""
    detector = ScriptedDetector({0: [face(FACE)], 5: [face(FACE)], 10: [face(FACE)]})
    tracker = PiiTracker(detector)
    seen: dict[int, list[int]] = {}
    for frame in _static_frames(30):
        for result in tracker.push(frame):
            seen[result.index] = [r.track_id for r in result.regions]
            # Mutating the tracker after release must not alter what was handed out.
            assert result.index not in {s.index for s in tracker._buffer}
    assert all(ids == [0] for ids in seen.values())


# -------------------------------------------------------- three misses, drop


def test_face_is_dropped_after_three_missed_looks() -> None:
    """Found on looks 0..2, then missed on 3, 4 and 5: masked through frame 24, dropped from frame 25."""
    detector = ScriptedDetector({0: [face(FACE)], 5: [face(FACE)], 10: [face(FACE)]})

    results, tracker, _ = _run(detector, _static_frames(40))

    painted = _painted(results)
    assert painted == list(range(25))
    assert tracker.stats.tracks_dropped_missed == 1


def test_missed_looks_are_counted_on_each_frame() -> None:
    """The painter needs to know how many looks missed the face, to grow the mask more each time."""
    detector = ScriptedDetector({0: [face(FACE)], 5: [face(FACE)], 10: [face(FACE)]})

    results, _, _ = _run(detector, _static_frames(40))

    missed = {r.index: r.regions[0].missed_looks for r in results if r.regions}
    assert all(missed[i] == 0 for i in range(15))
    assert all(missed[i] == 1 for i in range(15, 20))
    assert all(missed[i] == 2 for i in range(20, 25))


def test_one_miss_does_not_end_the_face() -> None:
    """A face missed on one look and found on the next keeps the same identity."""
    detector = ScriptedDetector({0: [face(FACE)], 5: [face(FACE)], 15: [face(FACE)], 20: [face(FACE)]})

    results, tracker, _ = _run(detector, _static_frames(30))

    assert tracker.stats.tracks_created == 1
    assert tracker.stats.tracks_dropped_missed == 0
    assert {r.track_id for res in results for r in res.regions} == {0}
    assert all(_regions(results, i)[0].missed_looks == 0 for i in range(15, 25))


def test_face_found_again_after_two_misses_keeps_its_identity() -> None:
    """Two misses in a row are not enough to drop it; the next look rejoins the same track."""
    found = {i: [face(FACE)] for i in (0, 5, 20, 25, 30, 35)}  # misses on the looks at 10 and 15 only
    detector = ScriptedDetector(found)

    results, tracker, _ = _run(detector, _static_frames(40))

    assert tracker.stats.tracks_created == 1
    assert tracker.stats.tracks_dropped_missed == 0
    assert _painted(results) == list(range(40))


# ------------------------------------------------------------------ matching


def test_overlap_match_takes_the_highest_pair_first() -> None:
    """Two detections that both touch face B: the one that overlaps B most gets B, the other gets A."""
    a = box_mask((10, 10, 50, 50))
    b = box_mask((40, 10, 80, 50))
    y = box_mask((10, 10, 45, 50))  # IoU 0.875 with A, 0.09 with B
    x = box_mask((30, 10, 70, 50))  # IoU 0.33 with A, 0.60 with B
    detector = ScriptedDetector({0: [face(a), face(b)], 5: [face(x), face(y)]})

    results, tracker, _ = _run(detector, _static_frames(12, center=None))

    assert tracker.stats.tracks_created == 2
    on_frame_5 = {r.track_id: r.mask for r in _regions(results, 5)}
    assert np.array_equal(on_frame_5[0], y)
    assert np.array_equal(on_frame_5[1], x)


def test_first_label_sticks() -> None:
    """A face keeps the label it was first found under."""
    detector = ScriptedDetector({0: [face(FACE, label="human face")], 5: [face(FACE, label="head")]})

    results, _, _ = _run(detector, _static_frames(12))

    assert {r.label for res in results for r in res.regions} == {"human face"}


def test_low_score_and_tiny_masks_are_ignored() -> None:
    """A detection under the score cutoff, or under the minimum mask size, is not a face."""
    tiny = box_mask((10, 10, 20, 20))  # 100 px, under 500
    detector = ScriptedDetector({0: [face(FACE, score=0.49), face(tiny, score=0.99)]})

    results, tracker, _ = _run(detector, _static_frames(12))

    assert _painted(results) == []
    assert tracker.stats.detections_kept == 0
    assert tracker.stats.detections_filtered == 2


def test_score_at_the_cutoff_is_kept() -> None:
    """The cutoff is 'at least 0.50'."""
    detector = ScriptedDetector({0: [face(FACE, score=0.50)]})

    results, _, _ = _run(detector, _static_frames(12))

    assert 0 in _painted(results)


# --------------------------------------------------------------------- warps


class _FakePropagator:
    """Returns a chosen transform, and records which anchor frame each warp started from."""

    def __init__(self, matrix: npt.NDArray[np.float64] | None) -> None:
        self._matrix = matrix
        self.calls: list[tuple[int, int]] = []  # (anchor frame index, target frame index)

    def prepare(self, gray: npt.NDArray[np.uint8], mask: npt.NDArray[np.bool_]) -> Anchor:  # noqa: ARG002
        return Anchor(gray=gray, points=np.empty((0, 1, 2), dtype=np.float32))

    def estimate(self, anchor: Anchor, target_gray: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64] | None:
        self.calls.append((int(anchor.gray[0, 0]), int(target_gray[0, 0])))
        return self._matrix


def _marked_frames(n: int, center: tuple[int, int] | None = CENTER) -> list[npt.NDArray[np.uint8]]:
    """Build static frames whose top-left pixel is the frame index, so a test can tell which frame it holds."""
    frames = _static_frames(n, center)
    for i, frame in enumerate(frames):
        frame[0, 0] = i
    return frames


def _scale_about(center: tuple[int, int], scale: float) -> npt.NDArray[np.float64]:
    cx, cy = center
    return np.array([[scale, 0.0, cx * (1 - scale)], [0.0, scale, cy * (1 - scale)]])


def test_every_warp_starts_from_the_last_real_look() -> None:
    """A warped mask is never warped again: each warp goes from the look's own frame to the target."""
    propagator = _FakePropagator(np.eye(2, 3))
    detector = ScriptedDetector({0: [face(FACE)], 5: [face(FACE)], 10: [face(FACE)]})

    _run(detector, _marked_frames(14), propagator=propagator)

    forward = [(a, t) for a, t in propagator.calls if t > a]
    assert forward
    # The anchor is always the most recent look at or before the target.
    assert all(a == (t // 5) * 5 for a, t in forward if t % 5 != 0)


def test_backward_paint_also_starts_from_the_look() -> None:
    """Painting back from a new look warps the look's mask onto earlier frames, not frame over frame."""
    propagator = _FakePropagator(np.eye(2, 3))
    detector = ScriptedDetector({10: [face(FACE)], 15: [face(FACE)]})

    _run(detector, _marked_frames(20), propagator=propagator)

    backward = [(a, t) for a, t in propagator.calls if t < a]
    assert backward
    assert all(a == 10 for a, _ in backward)
    assert {t for _, t in backward} == set(range(10))


def test_warp_that_grows_the_mask_over_three_times_is_discarded() -> None:
    """A transform that quadruples the area is thrown away; the previous mask is held."""
    propagator = _FakePropagator(_scale_about(CENTER, 2.0))  # area x4
    detector = ScriptedDetector({0: [face(FACE)], 5: [face(FACE)]})

    results, tracker, _ = _run(detector, _static_frames(10), propagator=propagator)

    for i in range(1, 5):
        region = _regions(results, i)[0]
        assert region.held, i
        assert np.array_equal(region.mask, FACE), i
    assert tracker.stats.warps_held_growth > 0


def test_warp_under_the_growth_limit_is_kept() -> None:
    """A 1.5x scale is 2.25x the area, under the limit of 3, so the warped mask is used."""
    propagator = _FakePropagator(_scale_about(CENTER, 1.5))
    detector = ScriptedDetector({0: [face(FACE)], 5: [face(FACE)]})

    results, tracker, _ = _run(detector, _static_frames(10), propagator=propagator)

    region = _regions(results, 2)[0]
    assert not region.held
    assert np.count_nonzero(region.mask) > 2 * np.count_nonzero(FACE)
    assert tracker.stats.warps_held_growth == 0


def test_failed_transform_holds_the_mask_instead_of_leaving_a_gap() -> None:
    """If tracking cannot find a transform, the face stays masked where it last was."""
    propagator = _FakePropagator(None)
    detector = ScriptedDetector({0: [face(FACE)], 5: [face(FACE)]})

    results, tracker, _ = _run(detector, _static_frames(10), propagator=propagator)

    for i in range(1, 5):
        region = _regions(results, i)[0]
        assert region.held
        assert np.array_equal(region.mask, FACE)
    assert tracker.stats.warps_held_no_transform > 0


# --------------------------------------------------- real Lucas-Kanade motion


def _moving_clip(n: int, velocity: tuple[int, int], pan: tuple[int, int]) -> tuple[list, list]:
    """Build a face drifting at ``velocity`` px/frame while the camera pans at ``pan`` px/frame."""
    scene = Scene()
    centers = [(CENTER[0] + velocity[0] * i, CENTER[1] + velocity[1] * i) for i in range(n)]
    shifts = [(pan[0] * i, pan[1] * i) for i in range(n)]
    frames = [
        scene.render((cx - sx, cy - sy), camera_shift=(sx, sy))
        for (cx, cy), (sx, sy) in zip(centers, shifts, strict=True)
    ]
    truth = [disc_mask((cx - sx, cy - sy)) for (cx, cy), (sx, sy) in zip(centers, shifts, strict=True)]
    return frames, truth


@pytest.mark.parametrize(
    ("velocity", "pan"),
    [((0, 0), (0, 0)), ((1, 0), (0, 0)), ((0, 0), (1, 1)), ((1, -1), (-1, 0))],
    ids=["still", "face-moves", "camera-pans", "both"],
)
def test_lucas_kanade_follows_a_moving_face_between_looks(velocity: tuple[int, int], pan: tuple[int, int]) -> None:
    """Between looks the mask follows real motion. Looks return ground truth; in-between frames must track it."""
    frames, truth = _moving_clip(30, velocity, pan)
    detector = ScriptedDetector(lambda i: [face(truth[i])] if i % 5 == 0 else [])

    results, tracker, _ = _run(detector, frames)

    assert tracker.stats.warps_held_no_transform == 0
    for i in range(30):
        regions = _regions(results, i)
        assert len(regions) == 1, i
        assert iou(regions[0].mask, truth[i]) > 0.85, (i, iou(regions[0].mask, truth[i]))


def test_lucas_kanade_paints_a_late_face_back_onto_earlier_moving_frames() -> None:
    """A face that is moving when first found is followed backward through the frames before the look."""
    frames, truth = _moving_clip(40, (1, 0), (0, 0))
    detector = ScriptedDetector(lambda i: [face(truth[i])] if i in (20, 25, 30) else [])

    results, _, _ = _run(detector, frames)

    for i in range(10, 20):
        regions = _regions(results, i)
        assert len(regions) == 1, i
        assert iou(regions[0].mask, truth[i]) > 0.7, (i, iou(regions[0].mask, truth[i]))


def test_a_face_that_leaves_the_frame_does_not_break_tracking() -> None:
    """The face walks off the right edge. Tracking keeps running and the face is eventually dropped."""
    frames, truth = _moving_clip(70, (4, 0), (0, 0))
    detector = ScriptedDetector(lambda i: [face(truth[i])] if i % 5 == 0 and truth[i].sum() >= 500 else [])

    results, tracker, _ = _run(detector, frames)

    assert len(results) == 70
    assert tracker.stats.tracks_dropped_missed == 1


def test_lucas_kanade_needs_texture_and_says_so() -> None:
    """On a flat patch there are no corners to follow, so no transform is returned."""
    flat = np.full((240, 320), 128, dtype=np.uint8)
    propagator = LKRigidPropagator(LKConfig())

    anchor = propagator.prepare(flat, FACE)

    assert propagator.estimate(anchor, flat) is None
