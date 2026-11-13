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

"""Step 3: one compressed mask archive per clip."""

import io
import zipfile

import numpy as np
import pytest

from cosmos_curator.pipelines.video.pii.mask_archive import MaskArchiveWriter, read_mask_archive
from tests.cosmos_curator.pipelines.video.pii._pii_test_utils import HEIGHT, WIDTH, disc_mask

_TIMESTAMPS = np.arange(10, dtype=np.int64) * 41_666_667


def _writer(info: dict | None = None) -> MaskArchiveWriter:
    return MaskArchiveWriter(WIDTH, HEIGHT, _TIMESTAMPS, info or {})


def test_round_trip_is_exact() -> None:
    """Every mask comes back bit for bit."""
    writer = _writer()
    masks = {i: disc_mask((100 + 7 * i, 90)) for i in (0, 3, 4, 9)}
    for index, mask in masks.items():
        writer.add(index, mask)

    archive = read_mask_archive(writer.finish())

    for index, mask in masks.items():
        assert np.array_equal(archive.mask(index), mask)


def test_frames_without_a_mask_read_as_empty() -> None:
    """A frame with no mask has no entry, and reads back as all False."""
    writer = _writer()
    writer.add(2, disc_mask((100, 100)))
    writer.add(5, None)
    writer.add(6, np.zeros((HEIGHT, WIDTH), dtype=bool))

    data = writer.finish()
    archive = read_mask_archive(data)

    assert not archive.mask(5).any()
    assert not archive.mask(6).any()
    assert archive.meta["frames_with_mask"] == [2]
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert [n for n in z.namelist() if n.startswith("masks/")] == ["masks/00000002.bin"]


def test_the_archive_is_one_compressed_file() -> None:
    """Many frames of a small blob come to far less than raw, in a single ZIP."""
    writer = _writer()
    for i in range(10):
        writer.add(i, disc_mask((100 + 3 * i, 90)))

    data = writer.finish()

    raw_bytes = 10 * WIDTH * HEIGHT // 8
    assert len(data) < raw_bytes / 5
    assert zipfile.is_zipfile(io.BytesIO(data))
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in z.infolist())


def test_metadata_carries_timestamps_geometry_and_audit_info() -> None:
    """The archive says what clip it belongs to and how it was made."""
    writer = _writer({"paint_mode": "none", "mask_growth": 1.25})
    writer.add(1, disc_mask((50, 50)))

    meta = read_mask_archive(writer.finish()).meta

    assert (meta["width"], meta["height"], meta["frame_count"]) == (WIDTH, HEIGHT, 10)
    assert meta["timestamps_ns"] == [int(t) for t in _TIMESTAMPS]
    assert meta["info"] == {"paint_mode": "none", "mask_growth": 1.25}


def test_wrong_shape_is_rejected() -> None:
    """A mask from a different frame size is a bug, not something to store."""
    with pytest.raises(ValueError, match="does not match"):
        _writer().add(0, np.ones((10, 10), dtype=bool))


def test_an_archive_with_no_masks_is_still_valid() -> None:
    """A clip with no faces gives an empty but readable archive."""
    archive = read_mask_archive(_writer().finish())

    assert archive.masks == {}
    assert not archive.mask(0).any()
