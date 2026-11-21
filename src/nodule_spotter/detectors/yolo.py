"""YOLOv5/v8 nodule detector backed by the ``ultralytics`` package.

This is a thin adapter. The heavy lifting is delegated to
``ultralytics.YOLO`` — we only translate its box outputs into the mask
format the MoE expects, optionally post-process (e.g. dilate, keep
top-K), and attach a per-box confidence so the clustering step can
prefer high-confidence evidence.

**Weights.** The paper uses a YOLO checkpoint fine-tuned on 2D lung CT
slices. Point ``weights_path`` at such a file — a stock COCO checkpoint
will not produce useful detections on CT.

**Shape assumption.** The detector takes a 2D array (H, W). It is
internally converted to a 3-channel uint8 image so Ultralytics can do
its usual letterbox + normalization. Boxes returned by the model are in
the same coordinate frame as the input.
"""
from __future__ import annotations

import logging
from typing import Literal, Sequence

import numpy as np

from nodule_spotter.detectors.base import BaseDetector, DetectorProposal

logger = logging.getLogger(__name__)


BoxMode = Literal["rect", "ellipse"]


class YoloNoduleDetector(BaseDetector):
    """Ultralytics-backed detector — emits one filled mask per box.

    Parameters
    ----------
    weights_path :
        Path to a trained ``.pt`` checkpoint. A lung-nodule fine-tuned
        YOLOv5s / v8s / v11n works out of the box. A COCO checkpoint
        will load but produce nothing useful on CT.
    name :
        Logical name used for provenance in ``MaskCandidate.source``.
        Default keeps it simple so each YOLO variant has its own slot
        in the MoE.
    conf_threshold :
        Minimum box confidence to keep. Ultralytics' default is 0.25.
    iou_threshold :
        NMS IoU threshold. Ultralytics default 0.45.
    max_det :
        Hard cap on proposals per slice.
    box_mode :
        ``"rect"`` fills the bounding box with 1s; ``"ellipse"`` fits an
        inscribed ellipse (closer to the typical round-ish nodule mask
        and often a tighter IoU against GT).
    device :
        ``"cpu"`` / ``"cuda"`` / ``"cuda:0"`` / ``None`` (let Ultralytics
        decide).
    class_ids :
        Detection class IDs to retain. Defaults to nodule class ``0``.
        Pass ``None`` to retain every class.
    """

    def __init__(
        self,
        weights_path: str,
        name: str = "yolo",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 50,
        box_mode: BoxMode = "ellipse",
        device: str | None = None,
        class_ids: Sequence[int] | None = (0,),
    ):
        if box_mode not in {"rect", "ellipse"}:
            raise ValueError("box_mode must be 'rect' or 'ellipse'")
        self.name = name
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.box_mode = box_mode
        self.device = device
        self.class_ids = (
            None if class_ids is None else frozenset(int(value) for value in class_ids)
        )
        self._model = None  # lazy — see warmup()

    def warmup(self) -> None:
        """Load the Ultralytics model once (fails fast on a bad path)."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YoloNoduleDetector requires the 'ultralytics' package. "
                "Install with `pip install ultralytics`."
            ) from e
        logger.info("loading YOLO weights from %s", self.weights_path)
        self._model = YOLO(self.weights_path)

    def detect(self, slice_image: np.ndarray) -> list[DetectorProposal]:
        if self._model is None:
            self.warmup()

        rgb = _to_rgb_uint8(slice_image)
        H, W = rgb.shape[:2]

        # Ultralytics accepts a numpy array directly; verbose=False keeps
        # it quiet when running a whole dataset.
        results = self._model.predict(  # type: ignore[union-attr]
            rgb,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        proposals: list[DetectorProposal] = []
        # Single-image predict returns a list of length 1.
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()       # (N, 4)
            conf = boxes.conf.cpu().numpy()       # (N,)
            box_classes = getattr(boxes, "cls", None)
            labels = (
                box_classes.cpu().numpy()
                if box_classes is not None
                else None
            )
            if self.class_ids is not None and labels is None:
                raise ValueError(
                    "Ultralytics result has no class labels, but class_ids "
                    "filtering is enabled."
                )
            # If the model carries segmentation masks (e.g. YOLOv8-seg),
            # prefer those — they match GT shape much better than a box.
            seg = getattr(res, "masks", None)
            seg_arr = None
            if seg is not None and seg.data is not None:
                seg_arr = seg.data.cpu().numpy()  # (N, H', W') in model-res
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                label = int(labels[i]) if labels is not None else None
                if self.class_ids is not None and label not in self.class_ids:
                    continue
                if seg_arr is not None and i < seg_arr.shape[0]:
                    mask = _resize_binary(seg_arr[i], (H, W))
                else:
                    mask = _box_to_mask(x1, y1, x2, y2, H, W, mode=self.box_mode)
                if mask.sum() == 0:
                    continue
                proposals.append(
                    DetectorProposal(
                        mask=mask,
                        confidence=float(conf[i]),
                        label=label,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        meta={"backend": "ultralytics"},
                    )
                )
        return proposals


# ── Helpers ───────────────────────────────────────────────────────────

def _to_rgb_uint8(slice_image: np.ndarray) -> np.ndarray:
    """Coerce a 2D float/uint8 slice into a 3-channel uint8 image."""
    arr = np.asarray(slice_image)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[-1] == 3:
            return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
        else:
            raise ValueError(f"unexpected slice shape: {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"expected 2D slice, got shape {arr.shape}")

    if arr.dtype != np.uint8:
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-6:
            arr = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr = ((arr - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def _box_to_mask(
    x1: float, y1: float, x2: float, y2: float,
    H: int, W: int, mode: BoxMode = "rect",
) -> np.ndarray:
    """Fill a bounding box as a solid mask, optionally as an ellipse."""
    xi1, yi1 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
    xi2, yi2 = int(min(W, np.ceil(x2))), int(min(H, np.ceil(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return np.zeros((H, W), dtype=np.uint8)

    mask = np.zeros((H, W), dtype=np.uint8)
    if mode == "rect":
        mask[yi1:yi2, xi1:xi2] = 1
        return mask

    # Inscribed ellipse centered in the box
    cy = 0.5 * (yi1 + yi2)
    cx = 0.5 * (xi1 + xi2)
    ry = 0.5 * (yi2 - yi1)
    rx = 0.5 * (xi2 - xi1)
    if rx <= 0 or ry <= 0:
        return mask
    yy, xx = np.ogrid[:H, :W]
    ell = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    mask[ell] = 1
    return mask


def _resize_binary(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize of a binary mask to ``target_shape``."""
    from PIL import Image
    m = (np.asarray(mask) > 0.5).astype(np.uint8) * 255
    pil = Image.fromarray(m).resize(
        (target_shape[1], target_shape[0]), resample=Image.NEAREST,
    )
    return (np.array(pil) > 0).astype(np.uint8)
