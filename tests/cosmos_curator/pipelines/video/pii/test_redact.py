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

"""Step 4: the whole engine on real MP4 bytes, ending in an encode that is checked before use.

Growth and kernel values here are arbitrary test inputs, not recommendations.
"""

import threading
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.pipelines.video.pii import decode as decode_module
from cosmos_curator.pipelines.video.pii.config import EncodeConfig, PaintConfig, PaintMode, RedactionConfig
from cosmos_curator.pipelines.video.pii.decode import FullRateDecoder
from cosmos_curator.pipelines.video.pii.encode import (
    RedactionError,
    encode_frames,
    resolve_bitrate,
    verify_encoded,
)
from cosmos_curator.pipelines.video.pii.mask_archive import read_mask_archive
from cosmos_curator.pipelines.video.pii.redact import redact_mp4
from tests.cosmos_curator.pipelines.video.pii._pii_test_utils import (
    Scene,
    ScriptedDetector,
    box_mask,
    disc_mask,
    encode_test_clip,
    face,
)

_NS_PER_MS = 1_000_000
_SAMPLE_CLIP = Path(__file__).parents[1] / "data" / "test_clip_10s.mp4"

# Arbitrary values that exercise the code. They are not defaults and not advice.
_GROWTH = 1.25
_MISSED = 1.5
_KERNEL = 31


def _config(mode: PaintMode = PaintMode.GAUSSIAN, **overrides: object) -> RedactionConfig:
    paint = PaintConfig(
        mode=mode,
        mask_growth=_GROWTH,
        missed_look_growth=_MISSED,
        blur_kernel=_KERNEL if mode is PaintMode.GAUSSIAN else None,
    )
    return RedactionConfig(paint=paint, **overrides)  # type: ignore[arg-type]


def _scene_clip(n: int = 40, timestamps_ms: list[int] | None = None) -> tuple[bytes, list[npt.NDArray[np.bool_]]]:
    """Build a face drifting right one pixel a frame, encoded to MP4, with its ground-truth masks."""
    scene = Scene()
    centers = [(120 + i, 120) for i in range(n)]
    frames = [scene.render(c) for c in centers]
    truth = [disc_mask(c) for c in centers]
    return encode_test_clip(frames, timestamps_ms), truth


def _truth_detector(truth: list[npt.NDArray[np.bool_]]) -> ScriptedDetector:
    return ScriptedDetector(lambda i: [face(truth[i])] if i % 5 == 0 else [])


def _decoded(mp4: bytes) -> list[npt.NDArray[np.uint8]]:
    return list(FullRateDecoder(mp4).frames())


# ------------------------------------------------------------------ gaussian


def test_gaussian_keeps_every_frame_and_every_timestamp() -> None:
    """Same number of frames, at the same uneven times, as went in."""
    timestamps_ms = [0, 40, 100, 140, 300, 340, 520, 560, 600, 640] + [700 + 41 * i for i in range(30)]
    mp4, truth = _scene_clip(40, timestamps_ms)

    result = redact_mp4(mp4, _truth_detector(truth), _config())

    out = FullRateDecoder(result.mp4_bytes)
    assert out.frame_count == 40
    np.testing.assert_allclose(out.timestamps_ns, np.array(timestamps_ms) * _NS_PER_MS, atol=12_000)
    assert result.frame_count == 40


def test_gaussian_blurs_the_face_and_leaves_the_rest() -> None:
    """Inside the true face the detail is gone. Far from it the picture matches an untouched re-encode."""
    mp4, truth = _scene_clip(40)
    baseline = _decoded(redact_mp4(mp4, ScriptedDetector({}), _config()).mp4_bytes)

    redacted = _decoded(redact_mp4(mp4, _truth_detector(truth), _config()).mp4_bytes)

    for i, (b, r) in enumerate(zip(baseline, redacted, strict=True)):
        assert r[truth[i]].std() < 0.5 * b[truth[i]].std(), i
        far = ~cv2.dilate(truth[i].astype(np.uint8), np.ones((121, 121), np.uint8)).astype(bool)
        assert np.abs(r[far].astype(int) - b[far].astype(int)).mean() < 3, i


def test_gaussian_writes_no_mask_archive() -> None:
    """The blurred clip is the output; the masks are not kept."""
    mp4, truth = _scene_clip(20)

    result = redact_mp4(mp4, _truth_detector(truth), _config(PaintMode.GAUSSIAN))

    assert result.mask_archive is None


def test_blur_follows_the_face_all_the_way_to_the_last_frame() -> None:
    """The frames still in the hold when the clip ends are blurred too."""
    mp4, truth = _scene_clip(40)
    baseline = _decoded(redact_mp4(mp4, ScriptedDetector({}), _config()).mp4_bytes)

    redacted = _decoded(redact_mp4(mp4, _truth_detector(truth), _config()).mp4_bytes)

    for i in (36, 37, 38, 39):
        assert redacted[i][truth[i]].std() < 0.5 * baseline[i][truth[i]].std(), i


# ---------------------------------------------------------------------- none


def test_none_leaves_the_pixels_exactly_as_an_untouched_encode() -> None:
    """The no-blur mode feeds the encoder the decoded frames unchanged."""
    mp4, truth = _scene_clip(30)
    untouched = redact_mp4(mp4, ScriptedDetector({}), _config(PaintMode.NONE))

    none = redact_mp4(mp4, _truth_detector(truth), _config(PaintMode.NONE))

    assert none.mp4_bytes == untouched.mp4_bytes


def test_none_writes_one_archive_with_the_grown_masks() -> None:
    """The archive holds what gaussian would have blurred: every true face pixel, and a bit more."""
    mp4, truth = _scene_clip(30)

    result = redact_mp4(mp4, _truth_detector(truth), _config(PaintMode.NONE))

    assert result.mask_archive is not None
    archive = read_mask_archive(result.mask_archive)
    assert archive.meta["frame_count"] == 30
    assert archive.meta["info"]["paint_mode"] == "none"
    assert archive.meta["info"]["mask_growth"] == _GROWTH
    for i in range(30):
        mask = archive.mask(i)
        covered = np.count_nonzero(mask & truth[i]) / np.count_nonzero(truth[i])
        assert covered > 0.95, i
        assert np.count_nonzero(mask) > 1.3 * np.count_nonzero(truth[i]), i


def test_none_archive_timestamps_match_the_clip() -> None:
    """The archive is keyed to the same per-frame times as the video."""
    timestamps_ms = [i * 40 + (i % 3) * 5 for i in range(30)]
    mp4, truth = _scene_clip(30, timestamps_ms)

    result = redact_mp4(mp4, _truth_detector(truth), _config(PaintMode.NONE))

    assert result.mask_archive is not None
    assert read_mask_archive(result.mask_archive).meta["timestamps_ns"] == [int(t) for t in result.timestamps_ns]


# ------------------------------------------------------------- decode / encode


def test_the_clip_is_decoded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """One pass over the frames feeds the detector, the tracker, the painter and the encoder."""
    mp4, truth = _scene_clip(30)
    calls = {"frames": 0}
    original = decode_module.FullRateDecoder.frames

    def counting(self: FullRateDecoder, window_frames: int | None = None):  # noqa: ANN202
        calls["frames"] += 1
        return original(self, window_frames)

    monkeypatch.setattr(decode_module.FullRateDecoder, "frames", counting)
    detector = _truth_detector(truth)

    redact_mp4(mp4, detector, _config())

    assert calls["frames"] == 1
    assert detector.calls == [0, 5, 10, 15, 20, 25, 29]


def test_a_decode_window_does_not_change_the_output() -> None:
    """Bounding memory with a window gives the same bytes."""
    mp4, truth = _scene_clip(30)

    whole = redact_mp4(mp4, _truth_detector(truth), _config())
    windowed = redact_mp4(mp4, _truth_detector(truth), _config(decode_window_frames=7))

    assert windowed.mp4_bytes == whole.mp4_bytes


def test_the_detector_thread_does_not_change_the_output() -> None:
    """Looks run ahead on their own thread, and the bytes match running them inline."""
    mp4, truth = _scene_clip(30)
    inline_detector = _truth_detector(truth)
    threaded_detector = _truth_detector(truth)

    inline = redact_mp4(mp4, inline_detector, _config(detector_lookahead_frames=0))
    threaded = redact_mp4(mp4, threaded_detector, _config(detector_lookahead_frames=10))

    assert threaded.mp4_bytes == inline.mp4_bytes
    assert threaded_detector.calls == inline_detector.calls == [0, 5, 10, 15, 20, 25, 29]


def test_looks_run_off_the_calling_thread_unless_lookahead_is_zero() -> None:
    """With a lookahead every look runs on the worker. With 0 they run where redact_mp4 was called."""
    mp4, truth = _scene_clip(20)

    class ThreadRecorder:
        def __init__(self) -> None:
            self.threads: set[int] = set()

        def detect(self, frame_rgb: npt.NDArray[np.uint8], frame_index: int) -> list:
            self.threads.add(threading.get_ident())
            return _truth_detector(truth).detect(frame_rgb, frame_index)

    ahead, inline = ThreadRecorder(), ThreadRecorder()
    redact_mp4(mp4, ahead, _config(detector_lookahead_frames=10))  # type: ignore[arg-type]
    redact_mp4(mp4, inline, _config(detector_lookahead_frames=0))  # type: ignore[arg-type]

    assert inline.threads == {threading.get_ident()}
    assert len(ahead.threads) == 1
    assert threading.get_ident() not in ahead.threads


def test_a_negative_lookahead_is_refused() -> None:
    """0 turns the thread off. Below that is a mistake."""
    with pytest.raises(ValueError, match="detector_lookahead_frames"):
        _config(detector_lookahead_frames=-1)


def test_failed_encode_raises_and_returns_nothing() -> None:
    """If the encoder cannot run, there is no result for the caller to swap in."""
    mp4, truth = _scene_clip(20)
    config = _config(encode=EncodeConfig(encoder="not_a_real_encoder"))

    with pytest.raises(Exception, match=r"(?i)encoder|codec"):
        redact_mp4(mp4, _truth_detector(truth), config)


def test_a_detector_that_fails_stops_the_run() -> None:
    """A model error must not become a clip that looks checked."""
    mp4, _ = _scene_clip(20)

    class Broken:
        def detect(self, frame_rgb: npt.NDArray[np.uint8], frame_index: int) -> list:  # noqa: ARG002
            msg = "model fell over"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="model fell over"):
        redact_mp4(mp4, Broken(), _config())  # type: ignore[arg-type]


def test_odd_dimensions_are_refused() -> None:
    """yuv420p cannot hold an odd width, so refuse instead of cropping a row of pixels."""
    frame = np.zeros((240, 321, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="even dimensions"):
        encode_frames(
            [frame],
            np.array([0], dtype=np.int64),
            width=321,
            height=240,
            frame_rate=Fraction(24),
            bitrate_bps=1_000_000,
            config=EncodeConfig(),
        )


def test_fewer_frames_than_timestamps_is_an_error() -> None:
    """A short frame stream must not be encoded as a shorter clip."""
    frames = [np.zeros((240, 320, 3), dtype=np.uint8)] * 3

    with pytest.raises(RedactionError, match="expected 5"):
        encode_frames(
            frames,
            np.arange(5, dtype=np.int64) * 41_000_000,
            width=320,
            height=240,
            frame_rate=Fraction(24),
            bitrate_bps=1_000_000,
            config=EncodeConfig(),
        )


def test_verification_catches_a_different_timeline() -> None:
    """The output is checked against the input's timestamps before anything may replace it."""
    mp4, truth = _scene_clip(20)
    result = redact_mp4(mp4, _truth_detector(truth), _config())

    verify_encoded(result.mp4_bytes, result.timestamps_ns, width=320, height=240)
    with pytest.raises(RedactionError, match="timestamps drift"):
        verify_encoded(result.mp4_bytes, result.timestamps_ns + 50_000_000, width=320, height=240)
    with pytest.raises(RedactionError, match="frames, expected"):
        verify_encoded(result.mp4_bytes, result.timestamps_ns[:-1], width=320, height=240)


def test_bitrate_follows_the_input_unless_told_otherwise() -> None:
    """Configured wins, then the input's measured rate, then a size-based estimate."""
    rate = Fraction(24)

    assert resolve_bitrate(EncodeConfig(bitrate_bps=5_000_000), 1_000_000, 10, rate) == 5_000_000
    assert resolve_bitrate(EncodeConfig(), 1_234_000, 10, rate) == 1_234_000
    assert resolve_bitrate(EncodeConfig(), 0, 1000, rate) == 1000 * 8 * 24


# -------------------------------------------------------------- real sample clip


def _detail(frame: npt.NDArray[np.uint8], mask: npt.NDArray[np.bool_]) -> float:
    """Fine detail under ``mask``: the variance of the Laplacian, which a blur flattens."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F)[mask].var())


def test_real_sample_clip_end_to_end() -> None:
    """The repo's 10 s, 24 fps, 854x480 clip: every frame back, and the close-up face at the end blurred.

    The clip is snow, a title card, a bearded man, then a girl's face for the last ~25 frames.
    The scripted detector reports a box on her face from frame 215 on.
    """
    mp4 = _SAMPLE_CLIP.read_bytes()
    face_box = box_mask((250, 90, 470, 340), shape=(480, 854))
    detector = ScriptedDetector(lambda i: [face(face_box)] if i >= 215 and i % 5 == 0 else [])

    result = redact_mp4(mp4, detector, _config())

    untouched = _decoded(redact_mp4(mp4, ScriptedDetector({}), _config()).mp4_bytes)
    out = _decoded(result.mp4_bytes)
    assert result.frame_count == len(untouched) == len(out) == 240
    np.testing.assert_allclose(result.timestamps_ns, FullRateDecoder(mp4).timestamps_ns)
    assert result.stats.tracks_confirmed == 1
    # Where the face is, the detail is gone.
    for i in (225, 235, 239):
        assert _detail(out[i], face_box) < 0.3 * _detail(untouched[i], face_box), i
    # Long before the face, nothing has been touched beyond ordinary encoder noise.
    for i in range(0, 100, 10):
        assert np.abs(out[i].astype(int) - untouched[i].astype(int)).mean() < 1.5, i
