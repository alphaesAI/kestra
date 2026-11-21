"""Mask Clustering — DBSCAN over IoU distance.

Paper §Mask Clustering:
    d(m_i, m_j) = 1 − IoU(m_i, m_j),    IoU = |m_i ∩ m_j| / |m_i ∪ m_j|

Masks from different detectors that overlap substantially are grouped,
per-cluster pixel-wise averaged, and binarized at 0.5. Detector outputs
that don't fit any dense neighborhood are labeled noise by DBSCAN and
dropped as outliers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from nodule_spotter.types import ClusterResult, MaskCandidate

logger = logging.getLogger(__name__)

_SKLEARN_MISSING_WARNED = False


# ── IoU / distance ────────────────────────────────────────────────────

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union between two binary masks of matching shape.

    Empty-vs-empty returns 0.0 — those pairs should already have been
    filtered upstream, but we don't want a division-by-zero to bubble
    up from DBSCAN's distance callback.
    """
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} vs {b.shape}")
    ab = np.logical_and(a, b).sum()
    au = np.logical_or(a, b).sum()
    if au == 0:
        return 0.0
    return float(ab) / float(au)


def mask_distance(a: np.ndarray, b: np.ndarray) -> float:
    """``d = 1 − IoU``. In [0, 1]; 0 means identical."""
    return 1.0 - mask_iou(a, b)


def pairwise_distance_matrix(masks: list[np.ndarray]) -> np.ndarray:
    """Full symmetric N×N distance matrix for DBSCAN (metric='precomputed')."""
    n = len(masks)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = mask_distance(masks[i], masks[j])
            D[i, j] = D[j, i] = d
    return D


# ── Clusterer ─────────────────────────────────────────────────────────

@dataclass
class MaskClusterer:
    """DBSCAN clustering + per-cluster averaging of binary masks.

    Attributes
    ----------
    eps :
        DBSCAN neighborhood radius in IoU-distance units. ``eps=0.5`` is
        equivalent to an IoU threshold of 0.5 (`τ = 1 − ε`). Smaller
        eps = stricter grouping.
    min_samples :
        DBSCAN MinPts. This counts proposals, not distinct experts.
    min_unique_experts :
        Minimum number of distinct detector sources required for a valid
        cluster. This is the cross-expert agreement constraint. The class
        default remains ``1`` for existing single-detector callers; MoE
        configurations should set it to ``2`` or more.
    binarize_threshold :
        After averaging member masks pixel-wise, threshold to a binary
        final mask at this value (paper uses 0.5).
    """

    eps: float = 0.5
    min_samples: int = 2
    min_unique_experts: int = 1
    binarize_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        if self.min_unique_experts < 1:
            raise ValueError("min_unique_experts must be >= 1")

    # ── Main entry point ────────────────────────────────────────────

    def cluster(self, candidates: list[MaskCandidate]) -> list[ClusterResult]:
        """Group overlapping candidate masks and return per-cluster results.

        Returns an empty list when there are no candidates. Single-
        candidate inputs become either:
          * one noise cluster (dropped upstream) when ``min_samples > 1``,
          * or one singleton cluster when ``min_samples == 1``.
        """
        if not candidates:
            return []

        masks = [c.mask for c in candidates]
        D = pairwise_distance_matrix(masks)
        labels = self._dbscan(D)

        clusters: dict[int, list[int]] = {}
        noise: list[int] = []
        for idx, lab in enumerate(labels):
            if lab == -1:
                noise.append(idx)
            else:
                clusters.setdefault(int(lab), []).append(idx)

        results: list[ClusterResult] = []
        for lab, indices in sorted(clusters.items()):
            members = [candidates[i] for i in indices]
            avg = self._average_and_binarize([m.mask for m in members])
            has_expert_consensus = (
                len({member.source for member in members})
                >= self.min_unique_experts
            )
            results.append(
                ClusterResult(
                    cluster_id=lab,
                    members=members,
                    averaged_mask=avg,
                    is_noise=not has_expert_consensus,
                )
            )

        for idx in noise:
            cand = candidates[idx]
            results.append(
                ClusterResult(
                    cluster_id=-1,
                    members=[cand],
                    averaged_mask=(cand.mask > 0).astype(np.uint8),
                    is_noise=True,
                )
            )
        return results

    # ── Internals ──────────────────────────────────────────────────

    def _dbscan(self, D: np.ndarray) -> np.ndarray:
        """Thin wrapper over sklearn's DBSCAN with a precomputed matrix.

        Falls back to a hand-rolled implementation when sklearn is not
        installed so the module stays importable in minimal environments.
        """
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            global _SKLEARN_MISSING_WARNED
            if not _SKLEARN_MISSING_WARNED:
                logger.info("sklearn not available — using fallback DBSCAN")
                _SKLEARN_MISSING_WARNED = True
            return _fallback_dbscan(D, self.eps, self.min_samples)

        model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="precomputed",
        )
        return model.fit_predict(D)

    def _average_and_binarize(self, masks: list[np.ndarray]) -> np.ndarray:
        """Per-pixel mean of binary masks, thresholded to uint8 {0, 1}."""
        stacked = np.stack([m.astype(np.float32) for m in masks], axis=0)
        mean = stacked.mean(axis=0)
        binary = (mean >= self.binarize_threshold).astype(np.uint8)
        return binary


# ── Fallback DBSCAN (no sklearn) ──────────────────────────────────────

def _fallback_dbscan(D: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Minimal DBSCAN for the precomputed-distance case.

    Same label convention as sklearn: cluster ids 0, 1, 2, ... for core
    members and their reachables, ``-1`` for noise. Good enough for the
    tens-of-masks we see per slice; don't use for large inputs.
    """
    n = D.shape[0]
    labels = -np.ones(n, dtype=np.int64)

    def neighbors(i: int) -> list[int]:
        return [j for j in range(n) if j != i and D[i, j] <= eps]

    cluster_id = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        N = neighbors(i)
        if len(N) + 1 < min_samples:
            # labels[i] stays -1 (noise); may be re-labeled if later
            # reached by another point's expansion.
            continue
        labels[i] = cluster_id
        seeds = list(N)
        while seeds:
            j = seeds.pop()
            if labels[j] == -1:
                labels[j] = cluster_id
                Nj = neighbors(j)
                if len(Nj) + 1 >= min_samples:
                    for k in Nj:
                        if labels[k] == -1 and k not in seeds:
                            seeds.append(k)
        cluster_id += 1
    return labels
