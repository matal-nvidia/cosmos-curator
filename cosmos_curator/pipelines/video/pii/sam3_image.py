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

"""Still-image SAM3 detector for one frame at a time.

This is the MicroAGI shape, not Curator's video SAM3. The image is encoded once
per frame. Each prompt is a separate question about that encoding, and the
prompt embeddings are computed once at setup, not once per frame. The model
does not remember faces from one frame to the next. Following a face across
frames stays in :class:`~cosmos_curator.pipelines.video.pii.tracker.PiiTracker`.

The frame is stretched to a square. It is not padded, so a wide frame is
squashed and the mask is stretched back afterwards. The square is 1008 px: the
``facebook/sam3`` checkpoint's own ``image_size`` and the processor default. The
rotary position embeddings are precomputed for 72 x 72 patches (1008 / 14). At
1120 (80 x 80 patches) the forward pass fails with a shape mismatch, so the
plan's 1120 is not usable with these weights.
"""

from pathlib import Path
from typing import Any, Protocol, cast

import cv2
import numpy as np
import numpy.typing as npt

from cosmos_curator.pipelines.video.pii.detector import DEFAULT_LABEL, Detection

# The checkpoint's own image_size. MicroAGI's 1120 belongs to SAM3.1 and crashes
# on facebook/sam3 (rotary embeddings are sized for 1008). A face is kept at a
# score of at least 0.50. The 500 px minimum stays in the tracker.
INPUT_SIZE = 1008
SCORE_THRESHOLD = 0.50
# Where the model's soft mask becomes a hard one. This is the model's own
# binarization, not the blur and not the 500 px rule.
MASK_THRESHOLD = 0.5


def stretch_to_square(frame_rgb: npt.NDArray[np.uint8], size: int) -> npt.NDArray[np.uint8]:
    """Stretch ``frame_rgb`` to ``size`` by ``size``. Corners stay corners; nothing is padded."""
    return np.asarray(cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_LINEAR), dtype=np.uint8)


def mask_to_frame(mask: npt.NDArray[np.bool_], height: int, width: int) -> npt.NDArray[np.bool_]:
    """Stretch a square mask back to ``height`` by ``width`` without blending pixels."""
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return np.asarray(resized, dtype=bool)


class _Segmenter(Protocol):
    """The model half of the detector, so tests can stand in for SAM3."""

    def prepare(self, prompts: list[str]) -> None:
        """Cache one text embedding per prompt."""
        ...

    def segment(
        self, square_rgb: npt.NDArray[np.uint8], min_score: float = 0.0
    ) -> list[tuple[npt.NDArray[np.bool_], float, str]]:
        """Return ``(mask, score, prompt)`` at or above ``min_score``, masks at the square's size."""
        ...


class Sam3ImageSegmenter:
    """Runs ``facebook/sam3`` as a still-image model.

    Text embeddings are cached by :meth:`prepare`. :meth:`segment` encodes the
    image once and asks each cached prompt in turn. The two forward passes run
    under bf16 autocast unless ``bf16`` is off. The weights stay fp32, and the
    outputs are cast back to fp32 before masks are cut.
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        processor: Any,  # noqa: ANN401
        input_size: int = INPUT_SIZE,
        *,
        bf16: bool = True,
    ) -> None:
        """Wrap an already loaded ``Sam3Model`` and ``Sam3Processor``."""
        self._model = model
        self._processor = processor
        self._input_size = input_size
        self._bf16 = bf16
        self._device = model.device
        # (prompt, text-encoder output, attention mask). The model reads
        # ``.pooler_output`` off the text output, so the whole output is kept.
        self._prompts: list[tuple[str, Any, Any]] = []

    def prepare(self, prompts: list[str]) -> None:
        """Encode each prompt once."""
        import torch  # noqa: PLC0415 -- the package imports without torch installed

        cached: list[tuple[str, Any, Any]] = []
        for prompt in prompts:
            tokens = _to_device(self._processor(text=prompt, return_tensors="pt"), self._device)
            with torch.no_grad():
                text = self._model.get_text_features(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                )
            cached.append((prompt, text, tokens["attention_mask"]))
        self._prompts = cached

    def segment(
        self, square_rgb: npt.NDArray[np.uint8], min_score: float = 0.0
    ) -> list[tuple[npt.NDArray[np.bool_], float, str]]:
        """Encode ``square_rgb`` once and ask every prepared prompt.

        Only masks scoring at least ``min_score`` leave the GPU. Each look returns
        200 candidate masks as ``int64``, so copying all of them costs more than
        the forward pass.
        """
        import torch  # noqa: PLC0415 -- the package imports without torch installed

        if not self._prompts:
            msg = "prepare() must be called before segment()"
            raise RuntimeError(msg)
        image = _to_device(
            self._processor(
                images=square_rgb,
                return_tensors="pt",
                size={"height": self._input_size, "width": self._input_size},
            ),
            self._device,
        )
        autocast = torch.autocast(device_type=torch.device(self._device).type, dtype=torch.bfloat16, enabled=self._bf16)
        with torch.inference_mode(), autocast:
            vision = self._model.get_vision_features(pixel_values=image["pixel_values"])
        found: list[tuple[npt.NDArray[np.bool_], float, str]] = []
        target = [(self._input_size, self._input_size)]
        for prompt, text, attention_mask in self._prompts:
            with torch.inference_mode(), autocast:
                outputs = self._model(vision_embeds=vision, text_embeds=text, attention_mask=attention_mask)
            _outputs_to_fp32(outputs)
            # threshold 0: the library keeps scores *strictly above* its threshold,
            # and the plan says at least 0.50. ``min_score`` applies that comparison.
            result = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=0.0,
                mask_threshold=MASK_THRESHOLD,
                target_sizes=target,
            )[0]
            for mask, score in zip(result["masks"], result["scores"], strict=True):
                value = float(score)
                if value >= min_score:
                    found.append((_as_mask(mask), value, prompt))
        return found


class Sam3ImageDetector:
    """A :class:`~cosmos_curator.pipelines.video.pii.detector.PiiDetector` backed by still-image SAM3.

    Construct it with a ``segmenter`` in tests. Without one, :meth:`setup` loads
    ``facebook/sam3`` from the Curator weights cache.
    """

    def __init__(
        self,
        prompts: list[str] | None = None,
        *,
        input_size: int = INPUT_SIZE,
        score_threshold: float = SCORE_THRESHOLD,
        segmenter: _Segmenter | None = None,
    ) -> None:
        """Store the prompts. Nothing is loaded until :meth:`setup`."""
        if prompts is None:
            prompts = [DEFAULT_LABEL]
        if not prompts:
            msg = "Sam3ImageDetector requires at least one prompt"
            raise ValueError(msg)
        if input_size < 1:
            msg = f"input_size must be >= 1, got {input_size}"
            raise ValueError(msg)
        self._prompts = list(prompts)
        self._input_size = input_size
        self._score_threshold = score_threshold
        self._segmenter = segmenter
        self._ready = False

    @property
    def prompts(self) -> list[str]:
        """The prompts asked on every frame, in order."""
        return list(self._prompts)

    def setup(self) -> None:
        """Load the model if one was not injected, then cache the prompt embeddings."""
        if self._segmenter is None:
            self._segmenter = load_sam3_image_segmenter(self._input_size)
        self._segmenter.prepare(self._prompts)
        self._ready = True

    def detect(self, frame_rgb: npt.NDArray[np.uint8], frame_index: int) -> list[Detection]:  # noqa: ARG002
        """Return faces on ``frame_rgb``. Masks are at the frame's own size."""
        if not self._ready or self._segmenter is None:
            msg = "setup() must be called before detect()"
            raise RuntimeError(msg)
        height, width = frame_rgb.shape[:2]
        square = stretch_to_square(frame_rgb, self._input_size)
        detections: list[Detection] = []
        for mask, score, label in self._segmenter.segment(square, self._score_threshold):
            if score < self._score_threshold:
                continue
            full = mask_to_frame(mask, height, width)
            if not full.any():
                continue
            detections.append(Detection(mask=full, score=score, label=label))
        return detections


def load_sam3_image_segmenter(input_size: int = INPUT_SIZE, model_dir: Path | None = None) -> Sam3ImageSegmenter:
    """Load ``facebook/sam3`` onto the GPU.

    ``model_dir`` defaults to the Curator weights cache. Pass a directory to load
    a copy elsewhere.

    Raises:
        RuntimeError: no CUDA GPU is available.
        FileNotFoundError: the weights have not been downloaded.

    """
    import torch  # noqa: PLC0415 -- the package imports without torch installed
    from transformers import Sam3Model, Sam3Processor  # noqa: PLC0415 -- only present in the sam3 stack

    from cosmos_curator.core.utils.model.model_utils import get_local_dir_for_weights_name  # noqa: PLC0415

    if not torch.cuda.is_available():
        msg = "the SAM3 image detector requires a CUDA GPU"
        raise RuntimeError(msg)
    if model_dir is None:
        model_dir = get_local_dir_for_weights_name("facebook/sam3")
    if not model_dir.exists():
        msg = f"SAM3 weights not found at {model_dir}. Download them before running the detector."
        raise FileNotFoundError(msg)
    processor = Sam3Processor.from_pretrained(model_dir, local_files_only=True)
    model = cast("Any", Sam3Model.from_pretrained(model_dir, local_files_only=True)).to("cuda")
    model.eval()
    return Sam3ImageSegmenter(model, processor, input_size)


def _outputs_to_fp32(outputs: Any) -> None:  # noqa: ANN401
    for name in ("pred_masks", "pred_logits", "pred_boxes", "presence_logits"):
        value = getattr(outputs, name, None)
        if value is not None and value.is_floating_point():
            setattr(outputs, name, value.float())


def _to_device(batch: Any, device: Any) -> Any:  # noqa: ANN401
    if hasattr(batch, "to"):
        return batch.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _as_mask(mask: Any) -> npt.NDArray[np.bool_]:  # noqa: ANN401
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    return np.asarray(mask, dtype=bool)
