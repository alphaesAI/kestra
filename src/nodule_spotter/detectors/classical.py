"""Classical CV detector — torch-free baseline for smoke-testing the pipeline.

This is *not* a clinically useful detector. It runs a global brightness
threshold followed by connected-components and returns the largest few
blobs. Its purpose is twofold:

1. Let the MoE → Clustering → Spotter pipeline be exercised end-to-end
   on real images without requiring a GPU, torch, or trained weights.
2. Give a deliberately weak signal so the Mask Clustering step has to
   handle real noisy inputs (many blobs, some overlapping, most wrong).

It uses PIL + NumPy only. When varying the threshold / kernel
parameters, two instances produce *different* candidate sets — perfect
for exercising the DBSCAN clustering without any ML stack installed.
"""
from __future__ import annotations

import numpy as np

from nodule_spotter.detectors.base import BaseDetector, DetectorProposal


class ClassicalBlobDetector(BaseDetector):
    """Threshold + connected-components blob detector.

    Parameters
    ----------
    percentile :
        Pixels above this brightness percentile are treated as
        foreground. 99.5 keeps only the very brightest regions — a
        reasonable starting point for dense lung nodules on a standard
        soft-tissue window.
    min_area, max_area :
        Drop blobs outside this pixel-area range. The Shenzhen nodules
        sit roughly in [80, 6000] px on the 512×512 slices we checked.
    top_k :
        Keep only the K largest blobs by area (after min/max filtering).
    dilation_iters :
        Number of 1-pixel binary dilations applied to each blob before
        emission. A couple of iterations make the mask a bit more
        forgiving when computing IoU against a ground-truth mask drawn
        on the soft edge of the lesion.
    """

    def __init__(
        self,
        name: str = "classical_blob",
        percentile: float = 99.5,
        min_area: int = 80,
        max_area: int = 6000,
        top_k: int = 5,
        dilation_iters: int = 1,
    ):
        self.name = name
        self.percentile = percentile
        self.min_area = min_area
        self.max_area = max_area
        self.top_k = top_k
        self.dilation_iters = dilation_iters

    def detect(self, slice_image: np.ndarray) -> list[DetectorProposal]:
        arr = _as_float01(slice_image)
        if arr.size == 0:
            return []
        thr = np.percentile(arr, self.percentile)
        fg = (arr >= thr).astype(np.uint8)
        if fg.sum() == 0:
            return []

        labels, n = _connected_components(fg)
        candidates: list[tuple[int, int, np.ndarray]] = []
        for lab in range(1, n + 1):
            comp = (labels == lab)
            area = int(comp.sum())
            if area < self.min_area or area > self.max_area:
                continue
            mask = comp.astype(np.uint8)
            for _ in range(self.dilation_iters):
                mask = _dilate_1px(mask)
            candidates.append((area, lab, mask))

        candidates.sort(key=lambda t: t[0], reverse=True)
        proposals: list[DetectorProposal] = []
        for area, _, mask in candidates[: self.top_k]:
            # A crude confidence: normalize area into [0, 1] against the
            # configured max_area. Plenty for the clustering / panel to
            # have something to weight.
            conf = min(1.0, area / float(self.max_area))
            proposals.append(DetectorProposal(mask=mask, confidence=conf))
        return proposals


# ── Internals ─────────────────────────────────────────────────────────

def _as_float01(slice_image: np.ndarray) -> np.ndarray:
    arr = np.asarray(slice_image)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"expected 2D slice, got shape {arr.shape}")
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _connected_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected CCL implemented in pure NumPy (union-find via two passes).

    Good enough for 512×512 masks. For bigger images use scipy.ndimage.
    """
    H, W = binary.shape
    labels = np.zeros((H, W), dtype=np.int32)
    parent: list[int] = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    next_label = 1
    for y in range(H):
        for x in range(W):
            if not binary[y, x]:
                continue
            up = labels[y - 1, x] if y > 0 else 0
            lf = labels[y, x - 1] if x > 0 else 0
            if up == 0 and lf == 0:
                labels[y, x] = next_label
                parent.append(next_label)
                next_label += 1
            elif up != 0 and lf == 0:
                labels[y, x] = up
            elif up == 0 and lf != 0:
                labels[y, x] = lf
            else:
                labels[y, x] = min(up, lf)
                union(up, lf)

    # Flatten
    remap: dict[int, int] = {0: 0}
    next_flat = 0
    for y in range(H):
        for x in range(W):
            v = labels[y, x]
            if v == 0:
                continue
            root = find(v)
            if root not in remap:
                next_flat += 1
                remap[root] = next_flat
            labels[y, x] = remap[root]
    return labels, next_flat


def _dilate_1px(mask: np.ndarray) -> np.ndarray:
    """1-pixel square dilation via array shifts (no scipy)."""
    m = mask.astype(bool)
    out = m.copy()
    out[:-1, :] |= m[1:, :]
    out[1:, :] |= m[:-1, :]
    out[:, :-1] |= m[:, 1:]
    out[:, 1:] |= m[:, :-1]
    return out.astype(np.uint8)
