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

"""The detector seam: one still image in, face masks out.

The tracker never imports a model. It calls whatever implements
:class:`PiiDetector`, which is how the tracker is tested against a fake detector
before a real one exists.
"""

from collections.abc import Sequence
from typing import Protocol

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.pipelines.video.pii.config import TrackerConfig

DEFAULT_LABEL = "human face"


@attrs.define(frozen=True, eq=False)
class Detection:
    """One detected region on one frame.

    Attributes:
        mask: full-frame ``(H, W)`` boolean mask.
        score: detector confidence, higher is surer.
        label: which prompt produced it. The detector config is a list of
            prompts, so this is carried even though v0 asks only one.

    """

    mask: npt.NDArray[np.bool_]
    score: float
    label: str = DEFAULT_LABEL


class PiiDetector(Protocol):
    """Anything that finds PII on a single frame."""

    def detect(self, frame_rgb: npt.NDArray[np.uint8], frame_index: int) -> list[Detection]:
        """Return the detections for ``frame_rgb``, the frame at ``frame_index`` of the clip."""
        ...


def filter_detections(detections: Sequence[Detection], config: TrackerConfig) -> list[Detection]:
    """Drop detections below the score cutoff or under the minimum mask size."""
    return [
        det
        for det in detections
        if det.score >= config.min_score and int(np.count_nonzero(det.mask)) >= config.min_mask_pixels
    ]
