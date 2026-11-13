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

"""Configuration for the PII redaction engine.

Two groups of numbers live here and they are different in kind:

* :class:`TrackerConfig` carries the values copied from MicroAGI's published
  PII-Blurring guide (look every 5th frame, seen twice, three missed looks,
  score 0.50, 500 pixels, 3x warp growth) plus the 15 frame hold from the same
  guide.
* :class:`PaintConfig` carries the values that are *not* decided yet: how far a
  mask grows and how strong the Gaussian blur is. They have no defaults on
  purpose. Nothing in this module picks them.
"""

import enum

import attrs


@attrs.define(frozen=True)
class TrackerConfig:
    """How often to look for faces and how a face is followed.

    Attributes:
        look_interval: run the detector on every ``look_interval``-th frame
            (frames 0, 5, 10, ...). The frames in between follow the mask.
        hold_frames: how many frames stay open to change. A newly found face is
            painted backward onto the frames still held, and a frame is only
            released once it is more than ``hold_frames`` behind the newest one.
        confirm_looks: a face is only blurred once it has been seen on this
            many looks in a row.
        first_looks_exempt: the first this-many looks of a clip skip the wait,
            so the start of a clip is not left clear.
        max_missed_looks: a confirmed face is dropped after this many looks in
            a row that do not find it.
        min_score: a detection below this score is discarded.
        min_mask_pixels: a detection with fewer mask pixels than this is discarded.
        max_warp_growth: a warp that grows the mask by more than this factor is
            discarded and the previous mask is held instead.
        min_match_overlap: a detection joins an existing face only if their
            intersection-over-union is strictly greater than this. ``0.0`` means
            any overlap at all.

    """

    look_interval: int = 5
    hold_frames: int = 15
    confirm_looks: int = 2
    first_looks_exempt: int = 2
    max_missed_looks: int = 3
    min_score: float = 0.50
    min_mask_pixels: int = 500
    max_warp_growth: float = 3.0
    min_match_overlap: float = 0.0

    def __attrs_post_init__(self) -> None:
        """Reject values the tracker cannot run with."""
        if self.look_interval < 1:
            msg = f"look_interval must be >= 1, got {self.look_interval}"
            raise ValueError(msg)
        if self.hold_frames < self.look_interval:
            # A face confirmed on the next look must still be able to reach back to
            # the look where it was first seen.
            msg = f"hold_frames ({self.hold_frames}) must be >= look_interval ({self.look_interval})"
            raise ValueError(msg)
        if self.confirm_looks < 1:
            msg = f"confirm_looks must be >= 1, got {self.confirm_looks}"
            raise ValueError(msg)
        if self.first_looks_exempt < 0:
            msg = f"first_looks_exempt must be >= 0, got {self.first_looks_exempt}"
            raise ValueError(msg)
        if self.max_missed_looks < 1:
            msg = f"max_missed_looks must be >= 1, got {self.max_missed_looks}"
            raise ValueError(msg)
        if self.max_warp_growth < 1.0:
            msg = f"max_warp_growth must be >= 1.0, got {self.max_warp_growth}"
            raise ValueError(msg)


@attrs.define(frozen=True)
class LKConfig:
    """Prototype constants for the Lucas-Kanade mask follower.

    These are implementation choices for this prototype. MicroAGI does not
    publish them, so they are not claimed to match MicroAGI.

    Attributes:
        max_features: corners tracked per face.
        feature_quality: relative corner quality passed to ``goodFeaturesToTrack``.
        feature_min_distance: minimum pixel distance between corners.
        window_size: Lucas-Kanade search window, pixels.
        pyramid_levels: Lucas-Kanade pyramid depth.
        min_features: fewest surviving corners (and RANSAC inliers) needed to
            trust a transform.

    """

    max_features: int = 100
    feature_quality: float = 0.01
    feature_min_distance: int = 5
    window_size: int = 21
    pyramid_levels: int = 3
    min_features: int = 4


class PaintMode(enum.StrEnum):
    """What is written into the clip where a face mask is."""

    GAUSSIAN = "gaussian"
    NONE = "none"


@attrs.define(frozen=True)
class PaintConfig:
    """How the tracked masks are turned into pixels. No numbers are defaulted.

    Attributes:
        mode: ``gaussian`` blurs under the grown mask and keeps no masks;
            ``none`` leaves the pixels alone and writes a mask archive.
        mask_growth: scale applied to a mask about its centroid. ``1.0`` means
            no growth, ``1.1`` means 10% larger (EgoBlur's
            ``--scale_factor_detections`` shape). The grown mask always contains
            the original.
        missed_look_growth: extra scale applied once per consecutive missed look,
            on top of ``mask_growth``, because the position is less certain.
        blur_kernel: Gaussian kernel size in pixels, odd. Required for
            ``gaussian`` and ignored for ``none``.

    """

    mode: PaintMode
    mask_growth: float
    missed_look_growth: float
    blur_kernel: int | None = None

    def __attrs_post_init__(self) -> None:
        """Reject values the painter cannot run with."""
        if self.mask_growth < 1.0:
            msg = f"mask_growth must be >= 1.0 (1.0 = no growth), got {self.mask_growth}"
            raise ValueError(msg)
        if self.missed_look_growth < 1.0:
            msg = f"missed_look_growth must be >= 1.0, got {self.missed_look_growth}"
            raise ValueError(msg)
        if self.mode is PaintMode.GAUSSIAN:
            if self.blur_kernel is None:
                msg = "blur_kernel is required for gaussian mode and has no default"
                raise ValueError(msg)
            if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
                msg = f"blur_kernel must be a positive odd integer, got {self.blur_kernel}"
                raise ValueError(msg)


@attrs.define(frozen=True)
class EncodeConfig:
    """How the redacted frames are written back to an MP4.

    Attributes:
        encoder: FFmpeg encoder name. ``libopenh264`` is what the classic
            transcode stage defaults to.
        bitrate_bps: target bitrate. ``None`` matches the input clip's measured
            bitrate, so the redacted clip is not silently larger or smaller.

    """

    encoder: str = "libopenh264"
    bitrate_bps: int | None = None

    def __attrs_post_init__(self) -> None:
        """Reject a non-positive bitrate."""
        if self.bitrate_bps is not None and self.bitrate_bps <= 0:
            msg = f"bitrate_bps must be positive, got {self.bitrate_bps}"
            raise ValueError(msg)


@attrs.define(frozen=True)
class RedactionConfig:
    """Everything :func:`redact_mp4` needs.

    Attributes:
        paint: required; it carries the open numbers.
        tracker: tracker values, MicroAGI's by default.
        lk: Lucas-Kanade prototype constants.
        encode: output encode settings.
        decode_window_frames: decode this many frames at a time to bound memory,
            or ``None`` to decode the clip in one batch.

    """

    paint: PaintConfig
    tracker: TrackerConfig = attrs.Factory(TrackerConfig)
    lk: LKConfig = attrs.Factory(LKConfig)
    encode: EncodeConfig = attrs.Factory(EncodeConfig)
    decode_window_frames: int | None = None

    def __attrs_post_init__(self) -> None:
        """Reject a non-positive decode window."""
        if self.decode_window_frames is not None and self.decode_window_frames < 1:
            msg = f"decode_window_frames must be >= 1, got {self.decode_window_frames}"
            raise ValueError(msg)
