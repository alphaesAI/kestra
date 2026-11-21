"""Detection experts for the Nodule Spotter MoE."""
from __future__ import annotations

import importlib

from nodule_spotter.detectors.base import BaseDetector, DetectorProposal
from nodule_spotter.detectors.classical import ClassicalBlobDetector
from nodule_spotter.detectors.placeholders import (
    CenterPointNoduleDetector,
    DetrNoduleDetector,
    default_experts,
)

# Plugin registry for user-defined detector types.
_REGISTRY: dict[str, type] = {}


def register_detector(name: str):
    """Class decorator to register a custom detector under *name*.

    Example::

        @register_detector("my_detector")
        class MyDetector(BaseDetector):
            ...
    """
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def build_detector(spec: dict) -> BaseDetector:
    """Instantiate a detector from a config dict.

    The ``type`` key selects the implementation:
      - ``"classical"``   — :class:`ClassicalBlobDetector` (default)
      - ``"yolo"``        — :class:`YoloNoduleDetector`; requires ``weights_path``
      - ``"mmdet"``       — :class:`MMDetInferencer`; accepts either local
                            paths or a pinned Hugging Face artifact pair
      - ``"custom"``      — any importable class; requires ``class`` key
                            (e.g. ``"mypackage.submodule.MyDetector"``)
      - registered names  — any type added via :func:`register_detector`

    All remaining keys after popping ``type`` (and ``class``) are forwarded
    as constructor kwargs.
    """
    spec = dict(spec)
    dtype = spec.pop("type", "classical")

    if dtype == "classical":
        return ClassicalBlobDetector(**spec)

    if dtype == "yolo":
        from nodule_spotter.detectors.yolo import YoloNoduleDetector
        return YoloNoduleDetector(**spec)

    if dtype == "mmdet":
        from nodule_spotter.inference import MMDetInferencer
        return MMDetInferencer(**spec)

    if dtype == "custom":
        class_path = spec.pop("class")
        module_name, cls_name = class_path.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        return getattr(mod, cls_name)(**spec)

    if dtype in _REGISTRY:
        return _REGISTRY[dtype](**spec)

    raise ValueError(
        f"Unknown detector type {dtype!r}. "
        f"Built-in types: classical, yolo, mmdet, custom. "
        f"Registered: {sorted(_REGISTRY) or '(none)'}."
    )


# YoloNoduleDetector is exported lazily — imports ultralytics only when used.
def __getattr__(name: str):
    if name == "YoloNoduleDetector":
        from nodule_spotter.detectors.yolo import YoloNoduleDetector
        return YoloNoduleDetector
    raise AttributeError(name)


__all__ = [
    "BaseDetector",
    "CenterPointNoduleDetector",
    "ClassicalBlobDetector",
    "DetectorProposal",
    "DetrNoduleDetector",
    "YoloNoduleDetector",
    "build_detector",
    "default_experts",
    "register_detector",
]
