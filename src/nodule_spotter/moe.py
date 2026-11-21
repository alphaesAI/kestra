"""Mixture-of-Experts runner for the Nodule Spotter.

Paper §Mixture of Experts: *"These models operate in parallel, analyzing
individual lung CT slices independently, with each expert generating a
mask m to denote predicted nodule locations."*

This module drives that step. Given N :class:`BaseDetector` instances
and a CT slice, it runs every detector (off the event loop, via
``asyncio.to_thread`` since model forward passes are typically blocking
PyTorch code) and flattens the results into a single list of
:class:`MaskCandidate`. The output feeds directly into Mask Clustering.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Iterable

import numpy as np

from nodule_spotter.detectors.base import BaseDetector, DetectorProposal
from nodule_spotter.types import MaskCandidate

logger = logging.getLogger(__name__)


class MixtureOfExperts:
    """Runs a roster of detectors in parallel per CT slice.

    Parameters
    ----------
    detectors :
        The experts. At least one is required. Each must have a unique
        ``name`` — the name is preserved on every :class:`MaskCandidate`
        so downstream voting can tell "two detectors agreed" apart from
        "one detector fired twice".
    max_concurrency :
        Cap on how many detectors run concurrently. Most environments
        won't benefit from >N threads when N is the number of
        detectors, but a smaller cap helps when GPU memory is tight.
    warmup_on_init :
        Call ``detector.warmup()`` once at construction time. Useful to
        fail fast on missing weights.
    fail_fast :
        Propagate an expert failure by default. Set to ``False`` only when
        deliberately allowing inference with a partial expert roster.
    """

    def __init__(
        self,
        detectors: Iterable[BaseDetector],
        max_concurrency: int | None = None,
        warmup_on_init: bool = False,
        fail_fast: bool = True,
    ):
        self.detectors: list[BaseDetector] = list(detectors)
        if not self.detectors:
            raise ValueError("MixtureOfExperts requires at least one detector")
        names = [d.name for d in self.detectors]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Detector names must be unique; got duplicates in {names}"
            )
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = max_concurrency or len(self.detectors)
        self.fail_fast = fail_fast
        if warmup_on_init:
            for d in self.detectors:
                d.warmup()

    # ── Single-slice entry point ───────────────────────────────────────

    async def run_slice(
        self, slice_image: np.ndarray, slice_idx: int = 0
    ) -> list[MaskCandidate]:
        """Dispatch every detector on one slice, gather all proposals."""
        if slice_image.ndim != 2:
            # Allow (H, W, 1) too, but complain about anything else — most
            # clinical detectors want a 2D array and will misbehave silently
            # on a 3-channel image.
            if slice_image.ndim == 3 and slice_image.shape[-1] == 1:
                slice_image = slice_image[..., 0]
            else:
                raise ValueError(
                    f"slice_image must be 2D, got shape {slice_image.shape}"
                )

        sem = asyncio.Semaphore(self.max_concurrency)

        async def _run_one(det: BaseDetector) -> tuple[BaseDetector, list[DetectorProposal]]:
            async with sem:
                try:
                    proposals = await asyncio.to_thread(det.detect, slice_image)
                except Exception as e:  # noqa: BLE001
                    if self.fail_fast:
                        raise RuntimeError(
                            f"detector {det.name!r} failed on slice {slice_idx}"
                        ) from e
                    logger.warning(
                        "detector %r crashed on slice %d: %s",
                        det.name, slice_idx, e,
                    )
                    proposals = []
                return det, proposals

        pairs = await asyncio.gather(*[_run_one(d) for d in self.detectors])

        candidates: list[MaskCandidate] = []
        for det, proposals in pairs:
            for p in proposals:
                mask = _to_binary(p.mask, expected_shape=slice_image.shape)
                if mask is None:
                    continue
                candidates.append(
                    MaskCandidate(
                        mask=mask,
                        slice_idx=slice_idx,
                        source=det.name,
                        confidence=p.confidence,
                        meta={
                            **p.meta,
                            "label": p.label,
                            "bbox": p.bbox,
                        },
                    )
                )
        return candidates

    # ── Volume sweep ──────────────────────────────────────────────────

    async def run_volume(
        self, slices: Iterable[np.ndarray]
    ) -> list[list[MaskCandidate]]:
        """Run every slice sequentially, each with detectors in parallel.

        Inter-slice parallelism is deliberately left to the caller —
        stacking slices into a batch is model-specific and usually more
        efficient than naive async.
        """
        results: list[list[MaskCandidate]] = []
        for idx, sl in enumerate(slices):
            results.append(await self.run_slice(sl, slice_idx=idx))
        return results


# ── Internals ─────────────────────────────────────────────────────────

def _to_binary(mask: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray | None:
    """Coerce a detector output into a clean uint8 binary mask.

    Returns ``None`` if the mask is empty or the shape is incompatible
    with the slice — in either case we drop it rather than crash the
    whole MoE run.
    """
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.shape[:2] != expected_shape[:2]:
        logger.warning(
            "mask shape %s does not match slice shape %s — dropping",
            arr.shape, expected_shape,
        )
        return None
    binary = (arr > 0).astype(np.uint8)
    if binary.sum() == 0:
        return None
    return binary
