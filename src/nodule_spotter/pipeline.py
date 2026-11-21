"""End-to-end Nodule Spotter — MoE → Clustering → Judging Panel.

Usage sketch::

    from nodule_spotter import (
        NoduleSpotter, MixtureOfExperts, MaskClusterer,
        JudgingPanel, VLMJudge,
    )
    from nodule_spotter.detectors import default_experts

    spotter = NoduleSpotter(
        moe=MixtureOfExperts(default_experts()),
        clusterer=MaskClusterer(eps=0.5, min_samples=2),
        panel=JudgingPanel(judges=[VLMJudge("gpt-4o", model=...)]),
    )
    out = await spotter.spot_slice(ct_slice, slice_idx=42)
    final_masks = out.final_masks
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np

from nodule_spotter.clustering import MaskClusterer
from nodule_spotter.moe import MixtureOfExperts
from nodule_spotter.types import JudgedMask, SpotterOutput

if TYPE_CHECKING:
    from nodule_spotter.judging_panel import JudgingPanel

logger = logging.getLogger(__name__)


def _cluster_confidence(judged: JudgedMask) -> float:
    """Mean detector confidence of cluster members; missing values are zero."""
    confidences = [
        member.confidence
        for member in judged.cluster.members
        if member.confidence is not None
    ]
    return float(np.mean(confidences)) if confidences else 0.0


def _cluster_area(judged: JudgedMask) -> int:
    return int(np.count_nonzero(judged.cluster.averaged_mask))


def _has_agreement(judged: JudgedMask) -> bool:
    return len({member.source for member in judged.cluster.members}) >= 2


def select_top_masks(
    judged: list[JudgedMask],
    max_masks: int | None,
) -> list[np.ndarray]:
    """Return accepted masks, optionally capped by cross-expert evidence."""
    accepted = [item for item in judged if item.accepted]
    if max_masks is None or max_masks <= 0 or len(accepted) <= max_masks:
        return [item.cluster.averaged_mask for item in accepted]

    agreed = [item for item in accepted if _has_agreement(item)]
    ranked = sorted(
        agreed or accepted,
        key=lambda item: (
            _cluster_confidence(item),
            item.score,
            _cluster_area(item),
        ),
        reverse=True,
    )
    return [item.cluster.averaged_mask for item in ranked[:max_masks]]


@dataclass
class NoduleSpotter:
    """Compose MoE + clustering + judging into one per-slice call.

    ``panel`` is optional. When omitted (``panel=None``), the LLM
    Judging step is skipped and every non-noise cluster is accepted
    straight through. Use this mode for offline evaluation of the
    detection + clustering components when VLM APIs aren't available.

    ``max_masks`` limits accepted outputs per slice. ``None`` (or a
    non-positive value) keeps every accepted mask.
    """

    moe: MixtureOfExperts
    clusterer: MaskClusterer
    panel: "JudgingPanel | None" = None
    max_masks: int | None = 1

    # ── Single slice ────────────────────────────────────────────────

    async def spot_slice(
        self, slice_image: np.ndarray, slice_idx: int = 0
    ) -> SpotterOutput:
        """Run the full pipeline on one CT slice."""
        candidates = await self.moe.run_slice(slice_image, slice_idx=slice_idx)
        if not candidates:
            return SpotterOutput(
                slice_idx=slice_idx,
                final_masks=[],
                judged=[],
                raw_candidates=[],
            )

        clusters = self.clusterer.cluster(candidates)

        if self.panel is not None:
            judged = await self.panel.evaluate_all(slice_image, clusters)
        else:
            # Panel-free mode: auto-accept non-noise clusters with an
            # empty vote list and a placeholder score. Noise clusters
            # (DBSCAN label -1) are still rejected — they represent
            # detections no peer agreed with.
            judged = [
                JudgedMask(
                    cluster=c,
                    votes=[],
                    score=0.0 if c.is_noise else 1.0,
                    accepted=not c.is_noise,
                )
                for c in clusters
            ]

        final_masks = select_top_masks(judged, self.max_masks)
        return SpotterOutput(
            slice_idx=slice_idx,
            final_masks=final_masks,
            judged=judged,
            raw_candidates=candidates,
        )

    # ── Volume sweep ────────────────────────────────────────────────

    async def spot_volume(
        self,
        slices: Iterable[np.ndarray],
        max_slice_concurrency: int = 1,
    ) -> list[SpotterOutput]:
        """Run on every slice of a CT volume.

        ``max_slice_concurrency`` > 1 pipelines slices through the panel
        in parallel. Keep it at 1 when GPU memory is tight — detectors
        are already parallel *within* a slice via the MoE.
        """
        slice_list = list(slices)
        sem = asyncio.Semaphore(max(1, max_slice_concurrency))

        async def _one(i: int, sl: np.ndarray) -> SpotterOutput:
            async with sem:
                return await self.spot_slice(sl, slice_idx=i)

        return list(
            await asyncio.gather(
                *[_one(i, sl) for i, sl in enumerate(slice_list)]
            )
        )
