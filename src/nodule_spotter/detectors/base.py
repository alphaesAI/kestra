"""Abstract base class for Nodule Spotter detection experts.

Every expert inside the :class:`MixtureOfExperts` implements this
interface. A detector takes a 2D CT slice (H×W grayscale array, float
or uint8) and returns zero or more 2D binary masks — one per proposed
nodule — plus an optional per-proposal confidence.

Concrete MMDetection and Ultralytics adapters implement this interface.
Third-party detectors can implement it directly or use the detector factory;
the MoE runner is independent of the model framework.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DetectorProposal:
    """One proposal in the backend-independent MoE format.

    ``label`` and ``bbox`` retain the backend prediction for auditing.
    Framework-specific values may be stored in ``meta``.
    """

    mask: np.ndarray        # binary, same H×W as the input slice
    confidence: float | None = None
    label: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    """Protocol every MoE expert must satisfy.

    Subclasses MUST set :attr:`name` (used for provenance in
    ``MaskCandidate.source``) and implement :meth:`detect`. The
    :meth:`warmup` hook is optional — use it to load weights once
    before a batch of slices.
    """

    name: str = "unnamed_detector"

    def warmup(self) -> None:
        """Load weights / allocate buffers. Called once before first use."""
        return None

    @abstractmethod
    def detect(self, slice_image: np.ndarray) -> list[DetectorProposal]:
        """Return zero or more proposals for nodules in ``slice_image``.

        Parameters
        ----------
        slice_image :
            2D array of shape (H, W). Pixel range is implementation-
            defined; most detectors will want to normalize to [0, 1] or
            apply a HU-window clip internally.

        Returns
        -------
        list[DetectorProposal]
            One :class:`DetectorProposal` per proposed nodule. The mask
            MUST be the same H×W as the input slice and MUST be binary
            (values in {0, 1}) or castable to bool.
        """
        raise NotImplementedError
