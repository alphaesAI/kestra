"""Core dataclasses shared across the Nodule Spotter pipeline.

A :class:`MaskCandidate` is one detector's proposal for a nodule on one
slice — a 2D binary mask plus the metadata needed to trace it back to
the specific detector and (optionally) a detector-side confidence.

Downstream, a :class:`ClusterResult` groups overlapping candidates
(via DBSCAN over ``1 - IoU``) into a single averaged mask, and a
:class:`SpotterOutput` is the validated per-slice result after the
Judging Panel has filtered false positives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# A mask is a 2D boolean / uint8 array with the same H×W as the slice.
# We keep the type loose here (numpy doesn't ship a satisfying alias) and
# rely on the helpers in clustering.py / image_utils.py to coerce shapes.
Mask = np.ndarray


@dataclass
class MaskCandidate:
    """One detector's proposal on one slice."""

    mask: Mask
    slice_idx: int
    source: str                    # detector name, e.g. "yolo_v5"
    confidence: float | None = None  # detector-side score, if any
    meta: dict[str, Any] = field(default_factory=dict)

    def area(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass
class ClusterResult:
    """One MoE cluster — a group of overlapping :class:`MaskCandidate` s.

    ``averaged_mask`` is the per-pixel mean of the members thresholded at
    0.5 (see paper §Mask Clustering). ``is_noise`` is True when DBSCAN
    marked the member as not belonging to any dense region; those are
    treated as outliers and dropped before validation.
    """

    cluster_id: int
    members: list[MaskCandidate]
    averaged_mask: Mask
    is_noise: bool = False

    def size(self) -> int:
        return len(self.members)

    def sources(self) -> list[str]:
        return [m.source for m in self.members]


@dataclass
class JudgeVote:
    """One VLM's verdict on one candidate mask.

    ``sign`` is +1 (approve) / -1 (reject), ``confidence`` lies in [0, 1].
    The Judging Panel aggregates via ``Σ sign_j · confidence_j``.
    """

    judge_name: str
    sign: int                # +1 approve, -1 reject
    confidence: float        # in [0, 1]
    raw_response: str = ""   # original VLM output, for debugging

    def weighted(self) -> float:
        return self.sign * self.confidence


@dataclass
class JudgedMask:
    """A cluster that has been evaluated by the Judging Panel."""

    cluster: ClusterResult
    votes: list[JudgeVote]
    score: float             # Σ sign_j · confidence_j
    accepted: bool           # score > 0


@dataclass
class SpotterOutput:
    """All accepted nodule masks for one CT slice."""

    slice_idx: int
    final_masks: list[Mask]                  # masks with score > 0
    judged: list[JudgedMask]                 # every cluster, accepted or not
    raw_candidates: list[MaskCandidate]      # every MoE proposal pre-clustering
