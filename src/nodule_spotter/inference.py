"""MMDetection adapter for the backend-independent Nodule Spotter API.

Artifacts can be supplied as a local config/checkpoint pair or resolved from
the Hugging Face Hub. Both resolution and model loading remain lazy.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from nodule_spotter.detectors.base import BaseDetector, DetectorProposal

logger = logging.getLogger(__name__)

BoxMode = Literal["rect", "ellipse"]


class MMDetInferencer(BaseDetector):
    """Adapt an MMDetection model to :class:`BaseDetector`.

    Model loading remains lazy: constructing this adapter never imports
    MMDetection or reads the checkpoint. The first call to :meth:`warmup` or
    :meth:`detect` resolves the configured artifacts and loads the model.
    """

    def __init__(
        self,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        name: str | None = None,
        score_thr: float = 0.3,
        box_mode: BoxMode = "ellipse",
        device: str = "cuda:0",
        class_ids: Sequence[int] | None = (0,),
        repo_id: str | None = None,
        config_file: str | None = None,
        checkpoint_file: str | None = None,
        revision: str | None = None,
        checkpoint_sha256: str | None = None,
        cache_dir: str | None = None,
    ):
        local_fields = (config_path, checkpoint_path)
        hub_fields = (repo_id, config_file, checkpoint_file, revision)
        uses_local = any(value is not None for value in local_fields)
        uses_hub = any(value is not None for value in hub_fields)
        if uses_local and uses_hub:
            raise ValueError(
                "Choose either local config_path/checkpoint_path or Hugging "
                "Face repo_id/config_file/checkpoint_file/revision, not both."
            )
        if uses_local and not all(local_fields):
            raise ValueError(
                "config_path and checkpoint_path are both required for local "
                "artifacts."
            )
        if uses_hub and not all(hub_fields):
            raise ValueError(
                "repo_id, config_file, checkpoint_file, and revision are all "
                "required for Hugging Face artifacts."
            )
        if not uses_local and not uses_hub:
            raise ValueError(
                "Provide either local config_path/checkpoint_path or Hugging "
                "Face repo_id/config_file/checkpoint_file/revision."
            )
        if box_mode not in {"rect", "ellipse"}:
            raise ValueError("box_mode must be 'rect' or 'ellipse'")
        if score_thr < 0:
            raise ValueError("score_thr must be >= 0")
        if checkpoint_sha256 is not None:
            digest = checkpoint_sha256.lower()
            if len(digest) != 64 or any(
                c not in "0123456789abcdef" for c in digest
            ):
                raise ValueError(
                    "checkpoint_sha256 must be a 64-character hexadecimal digest."
                )
            checkpoint_sha256 = digest

        self.config_path = str(config_path) if config_path is not None else None
        self.checkpoint_path = (
            str(checkpoint_path) if checkpoint_path is not None else None
        )
        self.repo_id = repo_id
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.revision = revision
        self.checkpoint_sha256 = checkpoint_sha256
        self.cache_dir = cache_dir
        source_name = self.config_path or self.config_file
        self.name = name or Path(str(source_name)).stem
        self.score_thr = float(score_thr)
        self.box_mode = box_mode
        self.device = device
        self.class_ids = (
            None if class_ids is None else tuple(int(value) for value in class_ids)
        )
        self._model = None

    @classmethod
    def from_lists(
        cls,
        configs: list[str],
        checkpoints: list[str],
        **kwargs,
    ) -> list["MMDetInferencer"]:
        """Create one adapter for each config/checkpoint pair."""
        if len(configs) != len(checkpoints):
            raise ValueError(
                "configs and checkpoints must contain the same number of items"
            )
        return [
            cls(config_path=config, checkpoint_path=checkpoint, **kwargs)
            for config, checkpoint in zip(configs, checkpoints)
        ]

    def warmup(self) -> None:
        """Load the configured model into memory."""
        if self._model is not None:
            return
        config_path, checkpoint_path = self._resolve_artifacts()
        init_detector, _ = _load_mmdet_apis()
        logger.info(
            "loading %s config=%s checkpoint=%s device=%s",
            self.name,
            config_path,
            checkpoint_path,
            self.device,
        )
        self._model = init_detector(
            config_path,
            checkpoint_path,
            device=self.device,
        )

    def detect(self, slice_image: np.ndarray) -> list[DetectorProposal]:
        """Run one CT slice and return framework-independent proposals."""
        if self._model is None:
            self.warmup()

        _, inference_detector = _load_mmdet_apis()
        rgb = _to_rgb_uint8(slice_image)
        height, width = rgb.shape[:2]
        result = inference_detector(self._model, rgb)
        return parse_mmdet_result(
            result,
            height=height,
            width=width,
            score_thr=self.score_thr,
            box_mode=self.box_mode,
            class_ids=self.class_ids,
        )

    def _resolve_artifacts(self) -> tuple[str, str]:
        """Resolve paths and verify the checkpoint without loading the model."""
        if self.config_path is None or self.checkpoint_path is None:
            assert self.repo_id is not None
            assert self.config_file is not None
            assert self.checkpoint_file is not None
            assert self.revision is not None
            logger.info(
                "resolving %s artifacts from %s@%s",
                self.name,
                self.repo_id,
                self.revision,
            )
            config_path = _hf_hub_download(
                repo_id=self.repo_id,
                filename=self.config_file,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
            checkpoint_path = _hf_hub_download(
                repo_id=self.repo_id,
                filename=self.checkpoint_file,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        else:
            config_path = self.config_path
            checkpoint_path = self.checkpoint_path

        for artifact in (config_path, checkpoint_path):
            if not Path(artifact).is_file():
                raise FileNotFoundError(f"Detection artifact not found: {artifact}")
        if self.checkpoint_sha256 is not None:
            verify_file_sha256(checkpoint_path, self.checkpoint_sha256)

        self.config_path = str(config_path)
        self.checkpoint_path = str(checkpoint_path)
        return self.config_path, self.checkpoint_path

    def __repr__(self) -> str:
        config_name = Path(str(self.config_path or self.config_file)).name
        return (
            f"MMDetInferencer(name={self.name!r}, "
            f"config={config_name!r}, "
            f"score_thr={self.score_thr}, device={self.device!r})"
        )


def _hf_hub_download(**kwargs) -> str:
    """Import the Hub client only when a remote artifact is first resolved."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Hugging Face artifacts require `huggingface-hub`. Install "
            "`nodule_spotter/requirements-detection.txt`."
        ) from exc
    return hf_hub_download(**kwargs)


def verify_file_sha256(path: str | Path, expected: str) -> None:
    """Raise when *path* does not match the expected SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != expected.lower():
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}."
        )


def _load_mmdet_apis():
    """Import the official MMDetection inference APIs only when required."""
    try:
        from mmdet.apis import inference_detector, init_detector
    except ImportError as exc:
        raise ImportError(
            "MMDetInferencer requires an installed MMDetection environment."
        ) from exc
    return init_detector, inference_detector


def parse_mmdet_result(
    result,
    *,
    height: int,
    width: int,
    score_thr: float,
    box_mode: BoxMode,
    class_ids: Sequence[int] | None = (0,),
) -> list[DetectorProposal]:
    """Convert an MMDetection ``DetDataSample`` without loading a model."""
    pred = getattr(result, "pred_instances", None)
    if pred is None:
        logger.warning(
            "MMDetection result has no pred_instances (type=%s)",
            type(result),
        )
        return []

    scores = _to_numpy(pred.scores)
    bboxes = _to_numpy(pred.bboxes)
    labels = _to_numpy(pred.labels) if hasattr(pred, "labels") else None
    allowed = None if class_ids is None else frozenset(int(v) for v in class_ids)
    if allowed is not None and labels is None:
        raise ValueError(
            "MMDetection result has no labels, but class_ids filtering is enabled."
        )

    segmentation_masks = None
    if getattr(pred, "masks", None) is not None:
        segmentation_masks = _to_numpy(pred.masks)

    proposals: list[DetectorProposal] = []
    for index, (score, box) in enumerate(zip(scores, bboxes)):
        score_value = float(score)
        if score_value < score_thr:
            continue

        label = int(labels[index]) if labels is not None else None
        if allowed is not None and label not in allowed:
            continue

        bbox = tuple(float(value) for value in box)
        if segmentation_masks is not None and index < len(segmentation_masks):
            mask = (segmentation_masks[index] > 0).astype(np.uint8)
            if mask.shape != (height, width):
                mask = _resize_mask(mask, height, width)
        else:
            mask = box_to_mask(
                *bbox,
                height=height,
                width=width,
                mode=box_mode,
            )
        if not np.any(mask):
            continue

        proposals.append(
            DetectorProposal(
                mask=mask,
                confidence=score_value,
                label=label,
                bbox=bbox,
                meta={"backend": "mmdet"},
            )
        )
    return proposals


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "to_ndarray"):
        return np.asarray(value.to_ndarray())
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _to_rgb_uint8(slice_image: np.ndarray) -> np.ndarray:
    """Convert a grayscale CT slice to the RGB uint8 form MMDet expects."""
    array = np.asarray(slice_image)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3 and array.shape[-1] == 3:
        return array if array.dtype == np.uint8 else array.astype(np.uint8)
    if array.ndim != 2:
        raise ValueError(f"expected 2D slice, got shape {array.shape}")

    if array.dtype != np.uint8:
        low, high = float(array.min()), float(array.max())
        if high - low < 1e-6:
            array = np.zeros_like(array, dtype=np.uint8)
        else:
            array = (
                (array - low) / (high - low) * 255.0
            ).clip(0, 255).astype(np.uint8)
    return np.stack([array, array, array], axis=-1)


def box_to_mask(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    height: int,
    width: int,
    mode: BoxMode = "ellipse",
) -> np.ndarray:
    """Rasterize a clipped rectangle or its inscribed ellipse."""
    if mode not in {"rect", "ellipse"}:
        raise ValueError("mode must be 'rect' or 'ellipse'")
    left = int(max(0, np.floor(x1)))
    top = int(max(0, np.floor(y1)))
    right = int(min(width, np.ceil(x2)))
    bottom = int(min(height, np.ceil(y2)))
    mask = np.zeros((height, width), dtype=np.uint8)
    if right <= left or bottom <= top:
        return mask
    if mode == "rect":
        mask[top:bottom, left:right] = 1
        return mask

    center_y = 0.5 * (top + bottom)
    center_x = 0.5 * (left + right)
    radius_y = 0.5 * (bottom - top)
    radius_x = 0.5 * (right - left)
    yy, xx = np.ogrid[:height, :width]
    mask[
        ((yy - center_y) / radius_y) ** 2
        + ((xx - center_x) / radius_x) ** 2
        <= 1.0
    ] = 1
    return mask


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    resized = image.resize((width, height), resample=Image.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8)
