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

"""Stage builder for PII redaction."""

import attrs

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.pipelines.video.pii.config import PaintMode

# From the Pennington and Cohen trials. Kernel 151 hid the face. Kernel 31 did not.
# Growth 1.25 and missed-look growth 1.5 are the values those runs used.
DEFAULT_MASK_GROWTH = 1.25
DEFAULT_MISSED_LOOK_GROWTH = 1.5
DEFAULT_BLUR_KERNEL = 151


@attrs.define(frozen=True)
class PiiRedactionStageConfig:
    """How the split pipeline asks the redaction stage to run.

    The three paint numbers default to the trial values. Pass another value to override one.
    """

    mask_growth: float = DEFAULT_MASK_GROWTH
    missed_look_growth: float = DEFAULT_MISSED_LOOK_GROWTH
    blur_kernel: int | None = DEFAULT_BLUR_KERNEL
    paint_mode: PaintMode = PaintMode.GAUSSIAN
    gpus_per_worker: float = 1.0
    verbose: bool = False


def build_pii_redaction_stages(config: PiiRedactionStageConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Construct the redaction stage. One worker per node, because it uses a GPU."""
    from cosmos_curator.pipelines.video.pii.stage import PiiRedactionStage  # noqa: PLC0415

    return [
        CuratorStageSpec(
            PiiRedactionStage(config),
            num_workers_per_node=1,
        ),
    ]
