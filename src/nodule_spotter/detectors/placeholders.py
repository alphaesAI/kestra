"""Placeholder detectors — interface-only stubs with no weights loaded.

Replace each ``detect`` body with a real model when ready. The
:class:`MixtureOfExperts` runner is already wired to accept any
:class:`BaseDetector` subclass, so plugging in a real implementation
is a one-file change.

Three stubs are provided on purpose — the paper uses N parallel experts
and the clustering/voting logic only becomes meaningful for N ≥ 2.
"""
from __future__ import annotations

import numpy as np

from nodule_spotter.detectors.base import BaseDetector, DetectorProposal


class DetrNoduleDetector(BaseDetector):
    """Stub for a DETR / Deformable-DETR style query-based detector."""

    name = "detr"

    def detect(self, slice_image: np.ndarray) -> list[DetectorProposal]:
        del slice_image  # TODO: plug in DETR weights
        return []


class CenterPointNoduleDetector(BaseDetector):
    """Stub for a CenterNet / dual-attention center-point detector.

    Corresponds to the UrRehman-style architecture cited in the paper.
    """

    name = "centerpoint"

    def detect(self, slice_image: np.ndarray) -> list[DetectorProposal]:
        del slice_image  # TODO: plug in CenterPoint weights
        return []


def default_experts() -> list[BaseDetector]:
    """Return the default three-expert roster used by the MoE.

    The YOLO expert lives in :mod:`nodule_spotter.detectors.yolo` and
    needs an ``ultralytics`` install plus a weights path, so it is NOT
    included by default. Build it explicitly::

        from nodule_spotter.detectors import YoloNoduleDetector
        experts = [YoloNoduleDetector("path/to/lung_nodule_yolov5.pt"), ...]
    """
    return [
        DetrNoduleDetector(),
        CenterPointNoduleDetector(),
    ]


# ── Tiny synthetic detector for wiring tests ──────────────────────────

class _DebugBlobDetector(BaseDetector):
    """Not a real model — emits one hard-coded rectangular mask.

    Useful only for smoke-testing the MoE → clustering → panel pipeline
    without any trained weights. The rectangle location is parameterised
    so two debug detectors can produce near-overlapping masks and a
    third can be wildly off, to exercise DBSCAN.
    """

    def __init__(
        self,
        name: str,
        cx: int,
        cy: int,
        w: int = 30,
        h: int = 30,
        confidence: float = 0.9,
    ):
        self.name = name
        self._cx = cx
        self._cy = cy
        self._w = w
        self._h = h
        self._confidence = confidence

    def detect(self, slice_image: np.ndarray) -> list[DetectorProposal]:
        H, W = slice_image.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        y0 = max(0, self._cy - self._h // 2)
        y1 = min(H, self._cy + self._h // 2)
        x0 = max(0, self._cx - self._w // 2)
        x1 = min(W, self._cx + self._w // 2)
        mask[y0:y1, x0:x1] = 1
        return [DetectorProposal(mask=mask, confidence=self._confidence)]
