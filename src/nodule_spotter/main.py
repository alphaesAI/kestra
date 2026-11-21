"""CLI entry point for running the Nodule Spotter on CT slices.

Typical usage after populating real detectors + a config file::

    python -m nodule_spotter.main \
        --slices-dir /path/to/ct_slices \
        --out-dir   ./spotter_out \
        --config    nodule_spotter/config.yaml

Outputs per slice:
  * ``<slice_stem>.json`` — per-cluster votes and accept/reject decisions
  * ``<slice_stem>_mask_<k>.png`` — one PNG per accepted final mask
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

import numpy as np  # noqa: E402

from nodule_spotter.clustering import MaskClusterer  # noqa: E402
from nodule_spotter.detectors import build_detector  # noqa: E402
from nodule_spotter.judging_panel import JudgingPanel, VLMJudge  # noqa: E402
from nodule_spotter.moe import MixtureOfExperts  # noqa: E402
from nodule_spotter.pipeline import NoduleSpotter  # noqa: E402
from nodule_spotter.settings import SpotterSettings  # noqa: E402
from nodule_spotter.types import SpotterOutput  # noqa: E402


# ── Slice IO ─────────────────────────────────────────────────────────

def load_slice(path: Path) -> np.ndarray:
    """Read a PNG / JPEG / NumPy slice into a 2D float array."""
    if path.suffix.lower() in {".npy"}:
        arr = np.load(path)
    else:
        from PIL import Image
        arr = np.array(Image.open(path).convert("L"))
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    return arr.astype(np.float32)


def save_mask(mask: np.ndarray, path: Path) -> None:
    from PIL import Image
    Image.fromarray((mask.astype(bool).astype(np.uint8) * 255)).save(path)


def trace(output: SpotterOutput) -> dict:
    """JSON-serializable summary of a SpotterOutput (no masks inlined)."""
    return {
        "slice_idx": output.slice_idx,
        "num_raw_candidates": len(output.raw_candidates),
        "candidate_sources": [c.source for c in output.raw_candidates],
        "num_clusters": len(output.judged),
        "num_accepted": len(output.final_masks),
        "judged": [
            {
                "cluster_id": j.cluster.cluster_id,
                "is_noise": j.cluster.is_noise,
                "sources": j.cluster.sources(),
                "score": j.score,
                "accepted": j.accepted,
                "votes": [
                    {
                        "judge": v.judge_name,
                        "sign": v.sign,
                        "confidence": v.confidence,
                    }
                    for v in j.votes
                ],
            }
            for j in output.judged
        ],
    }


# ── Runner ───────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    settings = SpotterSettings.load(args.config)
    print(settings.describe())

    detectors = [build_detector(spec) for spec in settings.detectors]
    moe = MixtureOfExperts(
        detectors,
        max_concurrency=settings.max_concurrency,
        fail_fast=settings.fail_fast,
    )
    clusterer = MaskClusterer(
        eps=settings.clustering.eps,
        min_samples=settings.clustering.min_samples,
        min_unique_experts=settings.clustering.min_unique_experts,
        binarize_threshold=settings.clustering.binarize_threshold,
    )
    judges = [
        VLMJudge(
            name=j.name,
            model=j.spec.litellm_model(temperature=j.temperature),
            input_mode=settings.panel.input_mode,  # type: ignore[arg-type]
        )
        for j in settings.panel.judges
    ]
    panel = (
        JudgingPanel(
            judges=judges,
            accept_threshold=settings.panel.accept_threshold,
        )
        if judges
        else None
    )
    spotter = NoduleSpotter(
        moe=moe,
        clusterer=clusterer,
        panel=panel,
        max_masks=settings.max_masks,
    )

    slices_dir = Path(args.slices_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slice_paths = sorted(
        p for p in slices_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".npy"}
    )
    if args.limit:
        slice_paths = slice_paths[: args.limit]

    print(f"Slices to process: {len(slice_paths)}")
    for idx, sp in enumerate(slice_paths):
        image = load_slice(sp)
        output = await spotter.spot_slice(image, slice_idx=idx)

        stem = sp.stem
        (out_dir / f"{stem}.json").write_text(
            json.dumps(trace(output), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        for k, mask in enumerate(output.final_masks):
            save_mask(mask, out_dir / f"{stem}_mask_{k}.png")

        print(
            f"[{idx+1}/{len(slice_paths)}] {sp.name}  "
            f"raw={len(output.raw_candidates)}  "
            f"clusters={len(output.judged)}  "
            f"accepted={len(output.final_masks)}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--slices-dir", required=True,
                   help="Directory of 2D CT slices (PNG / JPEG / .npy).")
    p.add_argument("--out-dir", default="./spotter_out")
    p.add_argument("--config", default=None,
                   help="Path to nodule_spotter/config.yaml.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N slices (for smoke tests).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run(args))
