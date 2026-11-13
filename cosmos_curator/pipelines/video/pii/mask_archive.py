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

"""One compressed archive of masks per clip, not one image per frame.

The archive is a ZIP of deflated entries: ``meta.json`` plus one ``masks/NNNNNNNN.bin``
per frame that has a mask. Each mask is a bit-packed boolean array, so a 1080p
mask is 260 KB before compression and a few hundred bytes after, since a face
is a small blob. A frame with no mask has no entry.

The writer keeps only compressed bytes, so a long clip does not hold every raw
mask in memory.
"""

import io
import json
import zipfile
from collections.abc import Mapping
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt

ARCHIVE_VERSION = 1
_META = "meta.json"
_MASK_PREFIX = "masks/"


class MaskArchiveWriter:
    """Accumulates per-frame masks into an in-memory compressed archive."""

    def __init__(self, width: int, height: int, timestamps_ns: npt.NDArray[np.int64], info: Mapping[str, Any]) -> None:
        """Start an archive for a clip of ``len(timestamps_ns)`` frames.

        ``info`` is free-form audit data (paint mode, growth, thresholds) stored in ``meta.json``.
        """
        self._width = width
        self._height = height
        self._timestamps_ns = [int(t) for t in timestamps_ns]
        self._info = dict(info)
        self._buffer = io.BytesIO()
        self._zip = zipfile.ZipFile(self._buffer, mode="w", compression=zipfile.ZIP_DEFLATED)
        self._frames_with_mask: list[int] = []

    def add(self, frame_index: int, mask: npt.NDArray[np.bool_] | None) -> None:
        """Store the mask for ``frame_index``. ``None`` or an empty mask stores nothing."""
        if mask is None or not mask.any():
            return
        if mask.shape != (self._height, self._width):
            msg = f"mask shape {mask.shape} does not match the clip's ({self._height}, {self._width})"
            raise ValueError(msg)
        self._zip.writestr(f"{_MASK_PREFIX}{frame_index:08d}.bin", np.packbits(mask).tobytes())
        self._frames_with_mask.append(frame_index)

    def finish(self) -> bytes:
        """Write ``meta.json`` and return the archive bytes."""
        meta = {
            "version": ARCHIVE_VERSION,
            "width": self._width,
            "height": self._height,
            "frame_count": len(self._timestamps_ns),
            "timestamps_ns": self._timestamps_ns,
            "frames_with_mask": self._frames_with_mask,
            "info": self._info,
        }
        self._zip.writestr(_META, json.dumps(meta))
        self._zip.close()
        return self._buffer.getvalue()


@attrs.define(eq=False)
class MaskArchive:
    """A loaded archive."""

    meta: dict[str, Any]
    masks: dict[int, npt.NDArray[np.bool_]]

    def mask(self, frame_index: int) -> npt.NDArray[np.bool_]:
        """Return the mask for ``frame_index``; all ``False`` for a frame that had none."""
        found = self.masks.get(frame_index)
        if found is not None:
            return found
        return np.zeros((self.meta["height"], self.meta["width"]), dtype=bool)


def read_mask_archive(data: bytes) -> MaskArchive:
    """Load an archive written by :class:`MaskArchiveWriter`."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        meta = json.loads(archive.read(_META))
        if meta["version"] != ARCHIVE_VERSION:
            msg = f"unsupported mask archive version {meta['version']}"
            raise ValueError(msg)
        pixels = meta["width"] * meta["height"]
        masks: dict[int, npt.NDArray[np.bool_]] = {}
        for frame_index in meta["frames_with_mask"]:
            packed = np.frombuffer(archive.read(f"{_MASK_PREFIX}{frame_index:08d}.bin"), dtype=np.uint8)
            masks[frame_index] = np.unpackbits(packed, count=pixels).astype(bool).reshape(meta["height"], meta["width"])
    return MaskArchive(meta=meta, masks=masks)
