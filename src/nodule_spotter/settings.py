"""Configuration loader for the Nodule Spotter module.

Keeps the same shape as ``lung_agents/settings.py`` so the two modules
can share a single ``config.yaml`` when run together, while remaining
independently loadable.

Expected config shape::

    defaults:
      api_base: https://your-openai-compatible-endpoint/v1
      api_key:  sk-replace-me
      model:    gpt-4o

    nodule_spotter:
      detectors:
        - type: classical
          name: smoke
      clustering:
        eps: 0.5
        min_samples: 2
        min_unique_experts: 2
        binarize_threshold: 0.5
      judging_panel:
        accept_threshold: 0.0
        input_mode: overlay
        judges:
          - name: gpt-4o
            model: gpt-4o
          - name: claude
            model: claude-sonnet-4-5
          - name: qwen-vl
            model: qwen2.5-vl-7b-instruct

Per-judge blocks inherit ``api_base`` / ``api_key`` from ``defaults``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


# ── Model endpoint ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    api_base: str
    api_key: str
    model: str

    def litellm_model_id(self) -> str:
        if "/" in self.model:
            return self.model
        return f"openai/{self.model}"

    def litellm_model(self, **extra: Any):
        from agentbase.model.litellm_model import LiteLLMModel

        return LiteLLMModel(
            self.litellm_model_id(),
            api_base=self.api_base,
            api_key=self.api_key,
            **extra,
        )


@dataclass(frozen=True)
class JudgeEntry:
    """Judge block from the config — a display name + its model spec."""

    name: str
    spec: ModelSpec
    temperature: float = 0.1


# ── Clustering / panel sections ──────────────────────────────────────

@dataclass
class ClusteringSettings:
    eps: float = 0.5
    min_samples: int = 2
    min_unique_experts: int = 2
    binarize_threshold: float = 0.5


@dataclass
class PanelSettings:
    accept_threshold: float = 0.0
    input_mode: str = "overlay"     # "overlay" | "pair"
    judges: list[JudgeEntry] = field(default_factory=list)


# ── Top-level ────────────────────────────────────────────────────────

@dataclass
class SpotterSettings:
    detectors: list[dict[str, Any]]
    clustering: ClusteringSettings
    panel: PanelSettings
    max_concurrency: int | None = None
    fail_fast: bool = True
    max_masks: int | None = 1
    raw: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SpotterSettings":
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"nodule_spotter config not found at {cfg_path}. "
                f"Copy config.example.yaml to config.yaml and fill in a key."
            )
        raw = _load_yaml(cfg_path)
        defaults = raw.get("defaults") or {}
        ns = raw.get("nodule_spotter") or {}

        clustering = ns.get("clustering") or {}
        panel_cfg = ns.get("judging_panel") or {}
        detectors = ns.get("detectors") or []
        if not isinstance(detectors, list) or not detectors:
            raise ValueError(
                "nodule_spotter.detectors must contain at least one detector spec."
            )
        if not all(isinstance(spec, dict) for spec in detectors):
            raise ValueError(
                "Every nodule_spotter.detectors entry must be a mapping."
            )

        # Resolve judges against defaults
        judge_entries: list[JudgeEntry] = []
        for j in panel_cfg.get("judges") or []:
            api_base = j.get("api_base") or defaults.get("api_base")
            api_key = j.get("api_key") or defaults.get("api_key")
            model = j.get("model") or defaults.get("model")
            missing = [
                k for k, v in (
                    ("api_base", api_base),
                    ("api_key", api_key),
                    ("model", model),
                ) if not v
            ]
            if missing:
                raise ValueError(
                    f"judge {j.get('name')!r} is missing fields: {missing}"
                )
            judge_entries.append(
                JudgeEntry(
                    name=j.get("name") or model,
                    spec=ModelSpec(api_base=api_base, api_key=api_key, model=model),
                    temperature=float(j.get("temperature", 0.1)),
                )
            )

        return cls(
            detectors=[dict(spec) for spec in detectors],
            clustering=ClusteringSettings(
                eps=float(clustering.get("eps", 0.5)),
                min_samples=int(clustering.get("min_samples", 2)),
                min_unique_experts=int(
                    clustering.get("min_unique_experts", 2)
                ),
                binarize_threshold=float(clustering.get("binarize_threshold", 0.5)),
            ),
            panel=PanelSettings(
                accept_threshold=float(panel_cfg.get("accept_threshold", 0.0)),
                input_mode=str(panel_cfg.get("input_mode", "overlay")),
                judges=judge_entries,
            ),
            max_concurrency=(
                int(ns["max_concurrency"])
                if ns.get("max_concurrency") is not None
                else None
            ),
            fail_fast=bool(ns.get("fail_fast", True)),
            max_masks=(
                int(ns["max_masks"])
                if ns.get("max_masks") is not None
                else None
            ),
            raw=raw,
            source_path=cfg_path,
        )

    # ── Diagnostics ──────────────────────────────────────────────────

    def describe(self) -> str:
        lines = [
            f"nodule_spotter settings ({self.source_path}):",
            "  detectors  : "
            f"{[spec.get('name') or spec.get('type') for spec in self.detectors]}",
            f"  clustering : eps={self.clustering.eps} "
            f"min_samples={self.clustering.min_samples} "
            f"min_unique_experts={self.clustering.min_unique_experts} "
            f"thresh={self.clustering.binarize_threshold}",
            f"  execution  : max_concurrency={self.max_concurrency} "
            f"fail_fast={self.fail_fast} max_masks={self.max_masks}",
            f"  panel      : accept>{self.panel.accept_threshold} "
            f"mode={self.panel.input_mode} "
            f"judges={[j.name for j in self.panel.judges]}",
        ]
        return "\n".join(lines)


# ── Internal helpers ─────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read nodule_spotter/config.yaml. "
            "Install with `pip install pyyaml`."
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a dict")
    return data
