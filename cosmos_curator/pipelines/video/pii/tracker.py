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

"""The tracker: decide, frame by frame, which masks apply.

Frames are pushed in order. Every ``look_interval``-th frame is a *look*: the
detector runs on it. The frames in between *follow* the masks found on the last
look. The last frame of a clip is also a look when it was not already one, so a
face that enters in the tail is still found. Frames are held for ``hold_frames``
before they are released, because a face found late is painted back onto the
frames still held.

The rules, all copied from MicroAGI's published behaviour:

* A detection joins the existing face it overlaps most (highest overlap first).
  The first label a face gets sticks.
* A new face is only blurred once a second look finds it, except on the first
  two looks of a clip and on the closing look. A face seen once and then missed
  is never blurred.
* A confirmed face that a look misses keeps its identity and its mask grows.
  After ``max_missed_looks`` misses in a row it is dropped.
* A new face is painted backward onto the held frames before it was found.
* Masks always move from the last real look, never from a warped mask.
* A warp that grows the mask beyond ``max_warp_growth`` is discarded, and the
  neighbouring frame's mask is held instead, so a bad warp never leaves a gap.

A frame is released (and so can be blurred and encoded) only once nothing later
can change it. The tracker only decides masks. It does not blur or encode.
"""

import collections
from collections.abc import Sequence

import attrs
import cv2
import numpy as np
import numpy.typing as npt

from cosmos_curator.pipelines.video.pii.config import TrackerConfig
from cosmos_curator.pipelines.video.pii.detector import Detection, PiiDetector, filter_detections
from cosmos_curator.pipelines.video.pii.propagate import Anchor, LKRigidPropagator, MaskPropagator, warp_mask


@attrs.define(eq=False)
class Region:
    """One face's mask on one frame.

    Attributes:
        track_id: which face. Stable for the life of the face.
        label: the prompt that first found it.
        mask: full-frame boolean mask, before any growth.
        missed_looks: consecutive looks that missed this face as of this frame.
            The painter grows the mask more the higher this is.
        held: ``True`` if the warp was unusable and a neighbouring mask was held.

    """

    track_id: int
    label: str
    mask: npt.NDArray[np.bool_]
    missed_looks: int
    held: bool = False


@attrs.define(eq=False)
class FrameResult:
    """A released frame and the masks that apply to it. Only confirmed faces appear."""

    index: int
    frame_rgb: npt.NDArray[np.uint8]
    regions: list[Region]


@attrs.define
class TrackerStats:
    """Counters over a whole clip, for audit and for tests."""

    frames: int = 0
    looks: int = 0
    detections_kept: int = 0
    detections_filtered: int = 0
    tracks_created: int = 0
    tracks_confirmed: int = 0
    tracks_dropped_missed: int = 0
    tracks_dropped_unconfirmed: int = 0
    warps: int = 0
    warps_held_no_transform: int = 0
    warps_held_growth: int = 0


@attrs.define(eq=False)
class _Track:
    track_id: int
    label: str
    confirmed: bool
    looks_seen: int
    missed_looks: int
    anchor_index: int
    anchor_mask: npt.NDArray[np.bool_]
    anchor_area: int
    anchor: Anchor


@attrs.define(eq=False)
class _Slot:
    index: int
    rgb: npt.NDArray[np.uint8]
    gray: npt.NDArray[np.uint8]
    regions: dict[int, Region] = attrs.Factory(dict)


def _iou(a: npt.NDArray[np.bool_], b: npt.NDArray[np.bool_]) -> float:
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    return int(np.count_nonzero(a & b)) / union


class PiiTracker:
    """Turns a stream of frames and a detector into per-frame masks."""

    def __init__(
        self,
        detector: PiiDetector,
        config: TrackerConfig | None = None,
        propagator: MaskPropagator | None = None,
    ) -> None:
        """Create a tracker. ``propagator`` defaults to rigid Lucas-Kanade."""
        self._detector = detector
        self._config = config or TrackerConfig()
        self._propagator: MaskPropagator = propagator or LKRigidPropagator()
        self._buffer: collections.deque[_Slot] = collections.deque()
        self._live: dict[int, _Track] = {}
        self._confirmed_ids: set[int] = set()
        self._next_index = 0
        self._next_track_id = 0
        self.stats = TrackerStats()

    # ------------------------------------------------------------------ public

    def push(self, frame_rgb: npt.NDArray[np.uint8]) -> list[FrameResult]:
        """Take the next frame. Returns the frames that are now final, oldest first."""
        index = self._next_index
        self._next_index += 1
        self.stats.frames += 1
        gray = np.asarray(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
        slot = _Slot(index=index, rgb=frame_rgb, gray=gray)
        self._buffer.append(slot)
        if index % self._config.look_interval == 0:
            self._look(slot)
        else:
            self._follow(slot)
        return self._release(keep_from=index - self._config.hold_frames)

    def finish(self) -> list[FrameResult]:
        """End of clip. Look at the last frame if it was not already a look, then release everything.

        A face found only on that closing look is blurred at once and painted
        backward through the hold. There is no later look that could confirm it.
        A face seen once on an earlier look and missed since then still stays
        unblurred.
        """
        if self._buffer:
            last = self._buffer[-1]
            if last.index % self._config.look_interval != 0:
                self._look(last, confirm_new=True)
        return self._release(keep_from=None)

    # -------------------------------------------------------------- per frame

    def _follow(self, slot: _Slot) -> None:
        """Carry each live face's mask forward from its last look onto a frame between looks."""
        previous = self._buffer[-2]
        for track in self._live.values():
            slot.regions[track.track_id] = self._project(
                track, slot, fallback=previous.regions[track.track_id].mask, missed_looks=track.missed_looks
            )

    def _look(self, slot: _Slot, *, confirm_new: bool = False) -> None:
        """Run the detector on a look frame, then match, confirm, miss, drop and create faces.

        ``confirm_new`` blurs a face that is first seen on this look, without
        waiting for another. The closing look of a clip sets it.
        """
        cfg = self._config
        self.stats.looks += 1
        look_number = self.stats.looks - 1

        raw = self._detector.detect(slot.rgb, slot.index)
        detections = filter_detections(raw, cfg)
        self.stats.detections_kept += len(detections)
        self.stats.detections_filtered += len(raw) - len(detections)

        previous = self._buffer[-2] if len(self._buffer) > 1 else None
        projected: dict[int, Region] = {}
        for track in self._live.values():
            fallback = previous.regions[track.track_id].mask if previous else track.anchor_mask
            projected[track.track_id] = self._project(track, slot, fallback=fallback, missed_looks=track.missed_looks)

        matches = self._match(detections, projected)
        matched_tracks = set(matches.values())

        for det_index, track_id in matches.items():
            self._accept(self._live[track_id], detections[det_index], slot)

        for track_id in list(self._live):
            if track_id in matched_tracks:
                continue
            track = self._live[track_id]
            if not track.confirmed:
                # Seen once, not found again: not a face we will blur.
                del self._live[track_id]
                self.stats.tracks_dropped_unconfirmed += 1
                continue
            track.missed_looks += 1
            if track.missed_looks >= cfg.max_missed_looks:
                del self._live[track_id]
                self.stats.tracks_dropped_missed += 1
                continue
            region = projected[track_id]
            region.missed_looks = track.missed_looks
            slot.regions[track_id] = region

        for det_index, detection in enumerate(detections):
            if det_index not in matches:
                self._create(detection, slot, exempt=confirm_new or look_number < cfg.first_looks_exempt)

    def _match(self, detections: Sequence[Detection], projected: dict[int, Region]) -> dict[int, int]:
        """Greedy match, highest overlap first. Returns ``{detection index: track id}``."""
        pairs: list[tuple[float, int, int]] = []
        for det_index, detection in enumerate(detections):
            for track_id, region in projected.items():
                overlap = _iou(detection.mask, region.mask)
                if overlap > self._config.min_match_overlap:
                    pairs.append((overlap, det_index, track_id))
        # Highest overlap first; ties break on detection then track so runs repeat exactly.
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        matches: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, det_index, track_id in pairs:
            if det_index in matches or track_id in used_tracks:
                continue
            matches[det_index] = track_id
            used_tracks.add(track_id)
        return matches

    def _accept(self, track: _Track, detection: Detection, slot: _Slot) -> None:
        """Re-anchor a face on the detection that found it again, and confirm it if that is its second look."""
        track.looks_seen += 1
        track.missed_looks = 0
        self._anchor(track, slot, detection.mask)
        if not track.confirmed and track.looks_seen >= self._config.confirm_looks:
            self._confirm(track)
        slot.regions[track.track_id] = Region(track.track_id, track.label, detection.mask, missed_looks=0)

    def _create(self, detection: Detection, slot: _Slot, *, exempt: bool) -> None:
        """Start a track for a face nobody was following, and paint it backward onto the held frames."""
        track = _Track(
            track_id=self._next_track_id,
            label=detection.label,
            confirmed=False,
            looks_seen=1,
            missed_looks=0,
            anchor_index=slot.index,
            anchor_mask=detection.mask,
            anchor_area=int(np.count_nonzero(detection.mask)),
            anchor=self._propagator.prepare(slot.gray, detection.mask),
        )
        self._next_track_id += 1
        self.stats.tracks_created += 1
        if exempt or self._config.confirm_looks <= 1:
            self._confirm(track)
        self._live[track.track_id] = track
        slot.regions[track.track_id] = Region(track.track_id, track.label, detection.mask, missed_looks=0)

        # Backward: newest first, each one's fallback is the frame after it.
        oldest_kept = slot.index - self._config.hold_frames
        later = slot
        for earlier in reversed(list(self._buffer)[:-1]):
            if earlier.index < oldest_kept:
                break
            earlier.regions[track.track_id] = self._project(
                track, earlier, fallback=later.regions[track.track_id].mask, missed_looks=0
            )
            later = earlier

    def _anchor(self, track: _Track, slot: _Slot, mask: npt.NDArray[np.bool_]) -> None:
        track.anchor_index = slot.index
        track.anchor_mask = mask
        track.anchor_area = int(np.count_nonzero(mask))
        track.anchor = self._propagator.prepare(slot.gray, mask)

    def _confirm(self, track: _Track) -> None:
        track.confirmed = True
        self._confirmed_ids.add(track.track_id)
        self.stats.tracks_confirmed += 1

    # ------------------------------------------------------------------ warps

    def _project(self, track: _Track, slot: _Slot, *, fallback: npt.NDArray[np.bool_], missed_looks: int) -> Region:
        """Return the face's mask on ``slot``, warped straight from its last real look.

        If the warp cannot be trusted, ``fallback`` (the neighbouring frame's mask)
        is held instead, so a face is never left unmasked because tracking failed.
        """
        self.stats.warps += 1
        matrix = self._propagator.estimate(track.anchor, slot.gray)
        if matrix is None:
            self.stats.warps_held_no_transform += 1
            return Region(track.track_id, track.label, fallback, missed_looks, held=True)
        warped = warp_mask(track.anchor_mask, matrix)
        growth = int(np.count_nonzero(warped)) / max(track.anchor_area, 1)
        if growth > self._config.max_warp_growth:
            self.stats.warps_held_growth += 1
            return Region(track.track_id, track.label, fallback, missed_looks, held=True)
        return Region(track.track_id, track.label, warped, missed_looks)

    # ---------------------------------------------------------------- release

    def _release(self, keep_from: int | None) -> list[FrameResult]:
        """Pop frames older than ``keep_from`` (all of them if ``None``)."""
        released: list[FrameResult] = []
        while self._buffer and (keep_from is None or self._buffer[0].index < keep_from):
            slot = self._buffer.popleft()
            regions = [slot.regions[tid] for tid in sorted(slot.regions) if tid in self._confirmed_ids]
            released.append(FrameResult(index=slot.index, frame_rgb=slot.rgb, regions=regions))
        return released
