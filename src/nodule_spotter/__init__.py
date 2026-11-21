"""Nodule Spotter — multi-expert nodule detection pipeline.

Implements the first module of the LungNoduleAgent paper:

    CT slices  ──►  Mixture of Experts (N detection models, parallel)
                 ──►  Mask Clustering (DBSCAN over 1 - IoU)
                 ──►  Judging Panel (VLM consensus, sign × confidence)
                 ──►  Final masks M

The public entry point is :class:`NoduleSpotter` in :mod:`.pipeline`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nodule_spotter.clustering import MaskClusterer  # noqa: E402
from nodule_spotter.detectors.base import BaseDetector  # noqa: E402
from nodule_spotter.moe import MixtureOfExperts  # noqa: E402
from nodule_spotter.types import (  # noqa: E402
    ClusterResult,
    JudgeVote,
    JudgedMask,
    Mask,
    MaskCandidate,
    SpotterOutput,
)

# JudgingPanel / VLMJudge / NoduleSpotter pull in EasyAgent + PIL; import
# them only via the public names below. Users who only need MoE +
# clustering (e.g. for offline debugging) don't pay that cost.
def __getattr__(name: str):  # noqa: D401
    if name in {"JudgingPanel", "VLMJudge"}:
        from nodule_spotter.judging_panel import JudgingPanel, VLMJudge
        return {"JudgingPanel": JudgingPanel, "VLMJudge": VLMJudge}[name]
    if name == "NoduleSpotter":
        from nodule_spotter.pipeline import NoduleSpotter
        return NoduleSpotter
    raise AttributeError(name)

__all__ = [
    "BaseDetector",
    "ClusterResult",
    "JudgeVote",
    "JudgedMask",
    "JudgingPanel",
    "Mask",
    "MaskCandidate",
    "MaskClusterer",
    "MixtureOfExperts",
    "NoduleSpotter",
    "SpotterOutput",
    "VLMJudge",
]
