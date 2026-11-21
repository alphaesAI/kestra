"""Detection-only evaluation harness for the Nodule Spotter pipeline.

This script runs MoE + Mask Clustering (and optionally the Judging
Panel) over a directory of CT slices and compares the final masks
against ground-truth masks, reporting per-image and aggregate IoU,
Dice and recall statistics.

Example — classical baseline, no LLM judge, 50 slices::

    python -m nodule_spotter.evaluate_detection \
        --images-dir  C:/…/shenzhen_processed/images \
        --masks-dir   C:/…/shenzhen_processed/masks \
        --limit 50 \
        --detector    classical

Swap in real YOLO weights::

    python -m nodule_spotter.evaluate_detection \
        --images-dir  C:/…/images --masks-dir C:/…/masks \
        --detector    yolo \
        --yolo-weights C:/…/lung_nodule_yolov5s.pt

Two YOLO variants + classical as a three-expert MoE::

    python -m nodule_spotter.evaluate_detection \
        --images-dir  C:/…/images --masks-dir C:/…/masks \
        --detector    yolo+yolo-seg+classical \
        --yolo-weights C:/…/lung_nodule_yolov5s.pt \
        --yolo-seg-weights C:/…/lung_nodule_yolov8s-seg.pt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

# Allow ``python -m nodule_spotter.evaluate_detection`` without installing.
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

from nodule_spotter.clustering import MaskClusterer  # noqa: E402
from nodule_spotter.detectors import (  # noqa: E402
    BaseDetector,
    ClassicalBlobDetector,
)
from nodule_spotter.moe import MixtureOfExperts  # noqa: E402
from nodule_spotter.pipeline import NoduleSpotter  # noqa: E402
from nodule_spotter.types import SpotterOutput  # noqa: E402

logger = logging.getLogger("nodule_spotter.eval")


# ── Metrics ───────────────────────────────────────────────────────────

@dataclass
class ImageMetrics:
    """Per-image evaluation summary."""

    stem: str
    gt_area: int
    num_final_masks: int
    best_iou: float
    best_dice: float
    union_iou: float           # IoU of GT vs. OR of all final masks
    union_dice: float
    hit: bool                  # best_iou >= iou_threshold
    num_candidates: int
    num_clusters: int

    def as_dict(self) -> dict:
        return {
            "stem": self.stem,
            "gt_area": self.gt_area,
            "num_final_masks": self.num_final_masks,
            "best_iou": round(self.best_iou, 4),
            "best_dice": round(self.best_dice, 4),
            "union_iou": round(self.union_iou, 4),
            "union_dice": round(self.union_dice, 4),
            "hit": self.hit,
            "num_candidates": self.num_candidates,
            "num_clusters": self.num_clusters,
        }


@dataclass
class AggregateMetrics:
    images: int = 0
    hits: int = 0
    gt_area_sum: int = 0
    best_iou_sum: float = 0.0
    union_iou_sum: float = 0.0
    best_dice_sum: float = 0.0
    union_dice_sum: float = 0.0
    empty_prediction: int = 0
    per_image: list[ImageMetrics] = field(default_factory=list)

    def add(self, m: ImageMetrics) -> None:
        self.images += 1
        self.gt_area_sum += m.gt_area
        self.best_iou_sum += m.best_iou
        self.union_iou_sum += m.union_iou
        self.best_dice_sum += m.best_dice
        self.union_dice_sum += m.union_dice
        self.hits += int(m.hit)
        if m.num_final_masks == 0:
            self.empty_prediction += 1
        self.per_image.append(m)

    def as_dict(self, iou_threshold: float) -> dict:
        n = max(1, self.images)
        return {
            "iou_threshold": iou_threshold,
            "images": self.images,
            "hits": self.hits,
            "recall": round(self.hits / n, 4),
            "empty_prediction": self.empty_prediction,
            "mean_best_iou": round(self.best_iou_sum / n, 4),
            "mean_union_iou": round(self.union_iou_sum / n, 4),
            "mean_best_dice": round(self.best_dice_sum / n, 4),
            "mean_union_dice": round(self.union_dice_sum / n, 4),
        }


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return 0.0 if union == 0 else inter / union


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    s = int(a.sum()) + int(b.sum())
    if s == 0:
        return 0.0
    return 2.0 * int(np.logical_and(a, b).sum()) / s


# ── Detector roster ──────────────────────────────────────────────────

def build_detectors(args: argparse.Namespace) -> list[BaseDetector]:
    """Turn ``--detector`` into a concrete list of BaseDetector instances.

    Accepted values (can be combined with ``+`` for MoE mode):
      * ``classical`` — torch-free blob detector (default)
      * ``classical2`` — a second classical detector with stricter params
        (for testing DBSCAN when no ML stack is available)
      * ``yolo``      — ``YoloNoduleDetector`` from ``--yolo-weights``
      * ``yolo-seg``  — a second YOLO using ``--yolo-seg-weights``
    """
    kinds = [k.strip() for k in args.detector.split("+") if k.strip()]
    detectors: list[BaseDetector] = []
    for k in kinds:
        if k == "classical":
            detectors.append(ClassicalBlobDetector(
                name="classical",
                percentile=args.classical_percentile,
                min_area=args.classical_min_area,
                max_area=args.classical_max_area,
                top_k=args.classical_top_k,
            ))
        elif k == "classical2":
            detectors.append(ClassicalBlobDetector(
                name="classical2",
                percentile=max(99.0, args.classical_percentile - 0.5),
                min_area=args.classical_min_area // 2,
                max_area=args.classical_max_area,
                top_k=max(1, args.classical_top_k - 2),
                dilation_iters=2,
            ))
        elif k in {"yolo", "yolo-seg"}:
            from nodule_spotter.detectors.yolo import YoloNoduleDetector
            weights = (
                args.yolo_weights if k == "yolo" else args.yolo_seg_weights
            )
            if not weights:
                raise SystemExit(
                    f"--detector includes {k!r} but "
                    f"--{'yolo-weights' if k == 'yolo' else 'yolo-seg-weights'} "
                    f"was not provided."
                )
            detectors.append(YoloNoduleDetector(
                weights_path=weights,
                name=k,
                conf_threshold=args.yolo_conf,
                box_mode=args.yolo_box_mode,  # type: ignore[arg-type]
                device=args.yolo_device,
            ))
        else:
            raise SystemExit(f"unknown --detector component: {k!r}")
    if not detectors:
        raise SystemExit("no detectors selected")
    return detectors


# ── IO ───────────────────────────────────────────────────────────────

def load_grayscale(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def load_binary_mask(path: Path) -> np.ndarray:
    """Load a mask as uint8 binary {0, 1}. Tolerates {0, 255} and {0, 1}."""
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 0).astype(np.uint8)


def union_of(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if not masks:
        return np.zeros(shape, dtype=np.uint8)
    acc = np.zeros(shape, dtype=bool)
    for m in masks:
        acc |= (m > 0)
    return acc.astype(np.uint8)


# ── Per-image evaluation ─────────────────────────────────────────────

def score_image(
    stem: str,
    gt_mask: np.ndarray,
    output: SpotterOutput,
    iou_threshold: float,
) -> ImageMetrics:
    H, W = gt_mask.shape

    if output.final_masks:
        best_iou = max(iou(gt_mask, m) for m in output.final_masks)
        best_dice = max(dice(gt_mask, m) for m in output.final_masks)
    else:
        best_iou = 0.0
        best_dice = 0.0

    union_mask = union_of(output.final_masks, (H, W))
    u_iou = iou(gt_mask, union_mask)
    u_dice = dice(gt_mask, union_mask)

    return ImageMetrics(
        stem=stem,
        gt_area=int(gt_mask.sum()),
        num_final_masks=len(output.final_masks),
        best_iou=best_iou,
        best_dice=best_dice,
        union_iou=u_iou,
        union_dice=u_dice,
        hit=best_iou >= iou_threshold,
        num_candidates=len(output.raw_candidates),
        num_clusters=len(output.judged),
    )


# ── Runner ───────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(
        p.stem for p in images_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if args.limit:
        stems = stems[: args.limit]

    detectors = build_detectors(args)
    print(f"Detectors: {[d.name for d in detectors]}")
    for d in detectors:
        try:
            d.warmup()
        except Exception as e:  # noqa: BLE001
            logger.warning("warmup failed for %s: %s", d.name, e)

    spotter = NoduleSpotter(
        moe=MixtureOfExperts(detectors),
        clusterer=MaskClusterer(
            eps=args.cluster_eps,
            min_samples=args.cluster_min_samples,
        ),
        panel=None,  # no VLM judge for offline detection eval
    )

    agg = AggregateMetrics()
    t0 = time.time()

    for idx, stem in enumerate(stems):
        img_path = images_dir / f"{stem}.png"
        msk_path = masks_dir / f"{stem}.png"
        if not msk_path.exists():
            logger.info("skipping %s — no GT mask", stem)
            continue

        image = load_grayscale(img_path)
        gt_mask = load_binary_mask(msk_path)
        if gt_mask.shape != image.shape:
            logger.warning(
                "shape mismatch on %s: img %s vs mask %s — skipping",
                stem, image.shape, gt_mask.shape,
            )
            continue

        output = await spotter.spot_slice(image, slice_idx=idx)
        m = score_image(stem, gt_mask, output, args.iou_threshold)
        agg.add(m)

        if args.save_predictions:
            pred = union_of(output.final_masks, gt_mask.shape)
            Image.fromarray(pred * 255).save(out_dir / f"{stem}_pred.png")

        if (idx + 1) % max(1, args.log_every) == 0 or idx == len(stems) - 1:
            running = agg.as_dict(args.iou_threshold)
            dt = time.time() - t0
            print(
                f"[{idx+1}/{len(stems)}] {stem} "
                f"best_iou={m.best_iou:.3f} dice={m.best_dice:.3f} "
                f"masks={m.num_final_masks} cands={m.num_candidates} "
                f"| running: recall={running['recall']:.3f} "
                f"mean_iou={running['mean_best_iou']:.3f} "
                f"elapsed={dt:.0f}s"
            )

    # ── Final report ──
    final = agg.as_dict(args.iou_threshold)
    print("\n" + "=" * 72)
    print("  DETECTION EVALUATION")
    print("=" * 72)
    for k, v in final.items():
        print(f"  {k:20s} {v}")

    report = {
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "aggregate": final,
        "per_image": [m.as_dict() for m in agg.per_image],
    }
    report_path = out_dir / "detection_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nReport: {report_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--masks-dir", required=True)
    p.add_argument("--out-dir", default="./nodule_spotter_eval")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N images (for smoke tests).")
    p.add_argument("--log-every", type=int, default=10)

    # Detector selection
    p.add_argument(
        "--detector", default="classical",
        help="Detector roster. Components joined with '+'. "
             "Options: classical, classical2, yolo, yolo-seg.",
    )
    p.add_argument("--classical-percentile", type=float, default=99.5)
    p.add_argument("--classical-min-area", type=int, default=80)
    p.add_argument("--classical-max-area", type=int, default=6000)
    p.add_argument("--classical-top-k", type=int, default=5)

    p.add_argument("--yolo-weights", default=None)
    p.add_argument("--yolo-seg-weights", default=None)
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-box-mode", choices=("rect", "ellipse"), default="rect")
    p.add_argument("--yolo-device", default=None)

    # Clustering
    p.add_argument("--cluster-eps", type=float, default=0.5)
    p.add_argument("--cluster-min-samples", type=int, default=1,
                   help="With a single detector, DBSCAN needs min_samples=1 "
                        "or every detection becomes noise. Raise to 2 when "
                        "running >=2 detectors in the MoE.")
    p.add_argument("--iou-threshold", type=float, default=0.3,
                   help="best_iou >= this counts as a 'hit'.")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run(args))
