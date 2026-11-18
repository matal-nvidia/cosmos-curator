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

"""Still-image SAM3 detector, with a stand-in for the model.

Nothing here loads ``facebook/sam3``. The tests check the wiring around it:
stretch without padding, one image encoding per frame however many prompts,
prompt embeddings cached once, every prompt in one decoder call, and the
score cutoff applied as "at least".
"""

import numpy as np
import pytest
import torch

from cosmos_curator.pipelines.video.pii import sam3_image
from cosmos_curator.pipelines.video.pii.sam3_image import (
    INPUT_SIZE,
    PRODUCTION_PROMPTS,
    Sam3ImageDetector,
    Sam3ImageSegmenter,
    load_sam3_image_segmenter,
    mask_to_frame,
    stretch_to_square,
)

_SIZE = 32  # a small square, so the tests do not allocate a 1008 image


class _Text:
    def __init__(self, prompt_id: int) -> None:
        self.pooler_output = torch.tensor([[prompt_id]], dtype=torch.float32)


class _Model:
    """Records how often the image and the prompts are encoded."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.vision_calls = 0
        self.text_calls: list[str] = []
        self.forwards: list[list[str]] = []

    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> _Text:  # noqa: ARG002
        prompt_id = int(input_ids[0, 0].item())
        self.text_calls.append(str(prompt_id))
        return _Text(prompt_id)

    def get_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.vision_calls += 1
        return pixel_values

    def __call__(self, vision_embeds: torch.Tensor, text_embeds: _Text, attention_mask: torch.Tensor) -> _Text:  # noqa: ARG002
        self.forwards.append([str(int(value)) for value in text_embeds.pooler_output[:, 0].tolist()])
        return text_embeds


class _Processor:
    def __init__(self, prompts: dict[str, int]) -> None:
        self._prompts = prompts
        self.image_sizes: list[tuple[int, int]] = []
        self.requested_size: list[dict[str, int]] = []

    def __call__(self, images: np.ndarray | None = None, text: str | None = None, **kwargs: object) -> dict:
        if images is not None:
            self.image_sizes.append(images.shape[:2])
            self.requested_size.append(kwargs["size"])  # type: ignore[arg-type]
            return {"pixel_values": torch.zeros(1, 3, images.shape[0], images.shape[1])}
        return {"input_ids": torch.tensor([[self._prompts[text or ""]]]), "attention_mask": torch.ones(1, 1)}

    def post_process_instance_segmentation(self, outputs: _Text, **kwargs: object) -> list[dict]:  # noqa: ARG002
        mask = torch.zeros(_SIZE, _SIZE)
        mask[: _SIZE // 2, : _SIZE // 4] = 1
        results = []
        for prompt_id in outputs.pooler_output[:, 0].tolist():
            score = 0.50 if int(prompt_id) == 1 else 0.49
            results.append({"masks": mask.unsqueeze(0), "scores": torch.tensor([score])})
        return results


def test_stretch_fills_the_square_and_pads_nothing() -> None:
    """A wide frame becomes a square, and both edges of the picture reach the edges of the square."""
    frame = np.zeros((10, 40, 3), dtype=np.uint8)
    frame[:, :20] = (255, 0, 0)
    frame[:, 20:] = (0, 0, 255)

    square = stretch_to_square(frame, 20)

    assert square.shape == (20, 20, 3)
    assert tuple(square[0, 0]) == (255, 0, 0)
    assert tuple(square[19, 19]) == (0, 0, 255)
    assert square[:, :, 1].max() == 0


def test_a_square_mask_maps_back_onto_the_wide_frame() -> None:
    """The left quarter of the square is the left quarter of the original frame."""
    mask = np.zeros((_SIZE, _SIZE), dtype=bool)
    mask[:, : _SIZE // 4] = True

    full = mask_to_frame(mask, 10, 40)

    assert full.shape == (10, 40)
    assert full[:, :10].all()
    assert not full[:, 10:].any()


def test_the_image_is_encoded_once_per_frame_and_the_prompts_once_per_run() -> None:
    """Two prompts on two frames: two image encodings, two prompt encodings, two decoder calls."""
    model = _Model()
    processor = _Processor({"human face": 1, "head": 2})
    segmenter = Sam3ImageSegmenter(model, processor, input_size=_SIZE)
    segmenter.prepare(["human face", "head"])
    frame = np.zeros((_SIZE, _SIZE, 3), dtype=np.uint8)

    segmenter.segment(frame)
    segmenter.segment(frame)

    assert model.vision_calls == 2
    assert model.text_calls == ["1", "2"]
    assert model.forwards == [["1", "2"], ["1", "2"]]
    assert processor.image_sizes == [(_SIZE, _SIZE), (_SIZE, _SIZE)]
    assert processor.requested_size == [{"height": _SIZE, "width": _SIZE}] * 2


def test_a_mask_under_the_cutoff_is_never_copied_off_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """0.50 is copied and returned. 0.49 is dropped before its mask is copied."""
    copied: list[object] = []
    real_as_mask = sam3_image._as_mask

    def counting_as_mask(mask: object) -> np.ndarray:
        copied.append(mask)
        return real_as_mask(mask)

    monkeypatch.setattr(sam3_image, "_as_mask", counting_as_mask)
    segmenter = Sam3ImageSegmenter(_Model(), _Processor({"human face": 1, "head": 2}), input_size=_SIZE)
    segmenter.prepare(["human face", "head"])

    found = segmenter.segment(np.zeros((_SIZE, _SIZE, 3), dtype=np.uint8), min_score=0.50)

    assert [(score, prompt) for _, score, prompt in found] == [(pytest.approx(0.50), "human face")]
    assert len(copied) == 1


def test_detect_rejects_a_score_under_the_cutoff_and_keeps_the_cutoff() -> None:
    """0.50 is kept. 0.49 is not. The mask comes back at the frame's size, under its own prompt."""
    model = _Model()
    processor = _Processor({"human face": 1, "head": 2})
    detector = Sam3ImageDetector(
        ["human face", "head"],
        input_size=_SIZE,
        segmenter=Sam3ImageSegmenter(model, processor, input_size=_SIZE),
    )
    detector.setup()
    frame = np.zeros((10, 40, 3), dtype=np.uint8)

    detections = detector.detect(frame, 0)

    assert [d.label for d in detections] == ["human face"]
    assert detections[0].score == pytest.approx(0.50)
    assert detections[0].mask.shape == (10, 40)
    assert detections[0].mask.any()


def test_detect_before_setup_is_refused() -> None:
    """A detector that has not loaded its prompts must not look like it found nothing."""
    detector = Sam3ImageDetector(["human face"], segmenter=Sam3ImageSegmenter(_Model(), _Processor({}), _SIZE))

    with pytest.raises(RuntimeError, match="setup"):
        detector.detect(np.zeros((4, 4, 3), dtype=np.uint8), 0)


def test_no_prompts_is_refused() -> None:
    """The prompt list can grow, but it cannot be empty."""
    with pytest.raises(ValueError, match="at least one prompt"):
        Sam3ImageDetector([])


def test_the_default_prompts_are_the_ten_from_the_guide() -> None:
    """No prompt list means the ten production prompts, each with its published score."""
    detector = Sam3ImageDetector()

    assert detector.prompts == [prompt for prompt, _floor in PRODUCTION_PROMPTS]
    assert dict(PRODUCTION_PROMPTS)["human face"] == 0.50
    assert dict(PRODUCTION_PROMPTS)["screen"] == 0.80
    assert INPUT_SIZE == 1008


def test_a_vaguer_prompt_keeps_its_higher_floor() -> None:
    """A screen at 0.60 is dropped. A face at 0.55 is kept. The floors are not one shared 0.50."""

    class _Fixed:
        def prepare(self, prompts: list[str]) -> None:
            self.prompts = prompts

        def segment(self, square_rgb: np.ndarray, min_score: float = 0.0) -> list[tuple[np.ndarray, float, str]]:
            assert min_score == pytest.approx(0.50)
            mask = np.zeros(square_rgb.shape[:2], dtype=bool)
            mask[0, 0] = True
            return [(mask, 0.60, "screen"), (mask, 0.55, "human face")]

    detector = Sam3ImageDetector(input_size=8, segmenter=_Fixed())
    detector.setup()

    detections = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8), 0)

    assert [d.label for d in detections] == ["human face"]


def test_loader_stops_when_the_weights_are_absent() -> None:
    """Loading must fail before any download when the facebook/sam3 weights are absent."""
    with pytest.raises(FileNotFoundError, match="weights not found"):
        load_sam3_image_segmenter()
