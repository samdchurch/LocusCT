#!/usr/bin/env python3
"""
Score the external VoxTell checkpoint on the official oncology held-out
test set (official_splits/onc_official_test_data.json), reporting Dice and
hit rate (dice >= 0.1) overall and per finding. Writes one GT/pred overlay
PNG per case (axial/coronal/sagittal, green GT contour, red predicted
contour) to <output's parent>/viz/<finding>/, matching
evaluate_voxtell_ed.py's style. Pass --no-visualize to skip this.

Unlike evaluate_onc_official_test.py (which scores our own model on the
352x352x180 resampled grid), this runs VoxTell on native-resolution NIfTI
volumes -- the format its "noResampling" nnU-Net plans expect -- paired
against native (pre-resample) masks. The manifest's image/mask relative
paths resolve unchanged against these native roots (resample_and_crop.py /
resample_masks.py preserve relative path structure when writing the
resampled copies), so the existing manifest is used as-is, just pointed at
different base directories. Because the two models run on different voxel
grids, these Dice numbers aren't a strictly apples-to-apples comparison
against evaluate_onc_official_test.py's resampled-grid numbers.

Oncology mask paths ("accession/mask_....nii.gz") don't embed a finding
category the way ED's ("CATEGORY/accession/Struct_....nii.gz") do, so
grouping uses each sample's "finding" field from the manifest instead of
parsing the mask path.

Samples sharing the same source image (multiple findings on one scan) are
batched into a single VoxTellPredictor call so the image is only encoded
once per scan, not once per finding.

Usage
-----
    python evaluate_voxtell_onc.py
    python evaluate_voxtell_onc.py --output outputs/eval/voxtell_onc/results.json
"""

import argparse
import json
import logging
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from tqdm import tqdm
from voxtell.inference.predictor import VoxTellPredictor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from utils.metrics import dice_score

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "onc_official_test_data.json"
DEFAULT_IMAGE_DIR = "/path/to/data/inhouse_abdominal_ct/nifti"
DEFAULT_MASK_DIR = "/path/to/data/inhouse_abdominal_ct/ALL_LABELS"
DEFAULT_MODEL_DIR = "/path/to/repo/voxtell/voxtell_v1.1"
DEFAULT_TEXT_ENCODER = "/path/to/data/models/Qwen3-Embedding-4B"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR,
                         help="Base dir the manifest's relative 'image' paths resolve against")
    parser.add_argument("--mask-dir", default=DEFAULT_MASK_DIR,
                         help="Base dir the manifest's relative 'mask' paths resolve against")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--text-encoding-model", default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--output", default="outputs/eval/voxtell_onc/results.json")
    parser.add_argument("--hit-threshold", type=float, default=0.1,
                         help="Dice value at/above which a sample counts as a 'hit'")
    parser.add_argument("--visualize", dest="visualize", action="store_true", default=False,
                         help="Write per-case GT/pred overlay PNGs (off by default)")
    parser.add_argument("--viz-output-dir", default=None,
                         help="Where to write visualization PNGs. Default: <output's parent>/viz")
    return parser.parse_args()


def _mask_centroid(mask: np.ndarray) -> tuple[int, int, int]:
    """(D, H, W) binary mask -> (d, h, w) centroid; falls back to volume center."""
    coords = np.argwhere(mask > 0.5)
    if len(coords) == 0:
        return tuple(s // 2 for s in mask.shape)
    d, h, w = coords.mean(axis=0).astype(int)
    d = int(np.clip(d, 0, mask.shape[0] - 1))
    h = int(np.clip(h, 0, mask.shape[1] - 1))
    w = int(np.clip(w, 0, mask.shape[2] - 1))
    return d, h, w


def _safe_filename(sample_id: str) -> str:
    name = sample_id.replace("/", "_").replace("\\", "_")
    for ext in (".nii.gz", ".nii", ".gz"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name[-150:] if len(name) > 150 else name


def _save_figure(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    sentence: str,
    dice: float,
    out_path: Path,
    vmin: float = -150.0,
    vmax: float = 250.0,
) -> None:
    d, h, w = _mask_centroid(gt_mask)

    def _slices(ci, cj, ck):
        return [
            ("Axial",    image[ci, :, :], gt_mask[ci, :, :], pred_mask[ci, :, :]),
            ("Coronal",  image[:, cj, :], gt_mask[:, cj, :], pred_mask[:, cj, :]),
            ("Sagittal", image[:, :, ck], gt_mask[:, :, ck], pred_mask[:, :, ck]),
        ]

    rows_views = [_slices(d, h, w)]
    row_labels = ["GT center"]

    pred_in_gt_slices = (
        pred_mask[d, :, :].any() or
        pred_mask[:, h, :].any() or
        pred_mask[:, :, w].any()
    )
    if not pred_in_gt_slices and pred_mask.any():
        pd, ph, pw = _mask_centroid(pred_mask)
        rows_views.append(_slices(pd, ph, pw))
        row_labels.append("Pred center")

    nrows = len(rows_views)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 5 * nrows), facecolor="black")
    fig.patch.set_facecolor("black")
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (views, row_label) in enumerate(zip(rows_views, row_labels)):
        for col_idx, (label, img_sl, gt_sl, pred_sl) in enumerate(views):
            ax = axes[row_idx, col_idx]
            img_sl = np.rot90(img_sl, 2)
            gt_sl = np.rot90(gt_sl, 2)
            pred_sl = np.rot90(pred_sl, 2)
            ax.imshow(img_sl, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal", origin="upper")
            if gt_sl.any():
                ax.contour(gt_sl, levels=[0.5], colors=["#00e676"], linewidths=1.5)
            if pred_sl.any():
                ax.contour(pred_sl, levels=[0.5], colors=["#ff1744"], linewidths=1.5)
            ax.set_title(label, color="white", fontsize=11, pad=4)
            ax.set_facecolor("black")
            ax.axis("off")
            if col_idx == 0 and nrows > 1:
                ax.text(0.02, 0.97, row_label, transform=ax.transAxes,
                        color="white", fontsize=9, va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

    legend = [
        Line2D([0], [0], color="#00e676", linewidth=1.5, label="GT"),
        Line2D([0], [0], color="#ff1744", linewidth=1.5, label="Pred"),
    ]
    axes[-1, -1].legend(handles=legend, loc="lower right", fontsize=9,
                        framealpha=0.4, facecolor="black", labelcolor="white",
                        edgecolor="gray")

    title = textwrap.fill(sentence, width=90) + f"\nDice: {dice:.3f}"
    fig.suptitle(title, color="white", fontsize=9, y=1.03, va="bottom")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def summarize(records: list[dict]) -> dict:
    dice_vals = [r["dice"] for r in records]
    return {
        "n_samples": len(records),
        "dice_mean": float(np.mean(dice_vals)) if dice_vals else 0.0,
        "dice_std": float(np.std(dice_vals)) if dice_vals else 0.0,
        "hit_rate": float(np.mean([r["hit"] for r in records])) if records else 0.0,
    }


def _write_summary(
    output_path: Path,
    model_dir: str,
    hit_threshold: float,
    records: list[dict],
    skipped: list[dict],
) -> dict:
    """Write the current summary to disk and return it. Called periodically during
    the run and again at the end (including on early exit), so a crash near the end
    of a multi-hour run can't wipe out everything computed so far."""
    by_category = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r)

    summary = {
        "model_dir": model_dir,
        "hit_threshold": hit_threshold,
        "overall": summarize(records),
        "by_category": {cat: summarize(recs) for cat, recs in sorted(by_category.items())},
        "n_skipped": len(skipped),
        "skipped": skipped,
        "per_sample": records,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = parse_args()

    with open(args.manifest) as f:
        samples = json.load(f)
    kept = [s for s in samples if s.get("sentence")]
    n_dropped = len(samples) - len(kept)
    if n_dropped:
        logger.warning(f"Dropping {n_dropped}/{len(samples)} sample(s) with no sentence: "
                        f"{[s['mask'] for s in samples if not s.get('sentence')]}")

    # Oncology mask paths ("accession/mask_....nii.gz") have no finding category
    # embedded in the path itself (unlike ED's "CATEGORY/accession/Struct_....nii.gz"),
    # so grouping comes from the manifest's own "finding" field instead.
    finding_lookup = {s["mask"]: (s.get("finding") or "Unknown") for s in kept}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = NibabelIOWithReorient()
    predictor = VoxTellPredictor(
        model_dir=args.model_dir,
        text_encoding_model=args.text_encoding_model,
        device=device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.visualize:
        viz_dir = Path(args.viz_output_dir) if args.viz_output_dir else output_path.parent / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing visualizations -> {viz_dir}")

    by_image: dict[str, list[dict]] = defaultdict(list)
    for s in kept:
        by_image[s["image"]].append(s)

    records: list[dict] = []
    skipped: list[dict] = []

    try:
        for group_idx, (image_rel, group) in enumerate(
            tqdm(sorted(by_image.items()), desc="Evaluating oncology official test set (VoxTell)")
        ):
            try:
                img, _ = reader.read_images([f"{args.image_dir}/{image_rel}"])
                sentences = [s["sentence"] for s in group]
                pred = predictor.predict_single_image(img, sentences)  # (n, X, Y, Z)
                image_np = img[0]
            except Exception as e:
                logger.warning(f"SKIP image group {image_rel}: {e}")
                skipped.extend({"image": image_rel, "mask": s["mask"], "reason": str(e)} for s in group)
                continue

            for i, s in enumerate(group):
                mask_rel = s["mask"]
                try:
                    seg, _ = reader.read_seg(f"{args.mask_dir}/{mask_rel}")
                    gt = (seg[0] > 0.5).astype(np.float32)
                    pr = pred[i].astype(np.float32)
                    if gt.shape != pr.shape:
                        raise ValueError(f"image/mask shape mismatch: pred={pr.shape} gt={gt.shape} "
                                          f"(mask likely drawn on a different reconstruction/series than "
                                          f"the manifest's 'image' field points to)")

                    dice = dice_score(
                        torch.from_numpy(pr[None]), torch.from_numpy(gt[None]),
                        threshold=0.5, smooth=1.0, from_logits=False,
                    )[0].item()

                    cat = finding_lookup.get(mask_rel, "Unknown")
                    records.append({
                        "id": mask_rel,
                        "category": cat,
                        "dice": dice,
                        "hit": bool(dice >= args.hit_threshold),
                    })

                    if args.visualize:
                        cat_dir = viz_dir / cat
                        cat_dir.mkdir(parents=True, exist_ok=True)
                        _save_figure(
                            image_np, gt, pr, s["sentence"], dice,
                            cat_dir / f"{_safe_filename(mask_rel)}.png",
                        )
                except Exception as e:
                    logger.warning(f"SKIP mask {mask_rel}: {e}")
                    skipped.append({"image": image_rel, "mask": mask_rel, "reason": str(e)})
                    continue

            if group_idx % 25 == 0:
                _write_summary(output_path, args.model_dir, args.hit_threshold, records, skipped)
    finally:
        summary = _write_summary(output_path, args.model_dir, args.hit_threshold, records, skipped)

        overall = summary["overall"]
        logger.info(
            f"Results -> {output_path}\n"
            f"  Overall ({overall['n_samples']} samples, {len(skipped)} skipped):\n"
            f"    Dice: {overall['dice_mean']:.4f} +/- {overall['dice_std']:.4f}   "
            f"Hit: {overall['hit_rate']:.4f}"
        )
        logger.info("  By finding:")
        for cat, s in summary["by_category"].items():
            logger.info(
                f"    {cat:<15} n={s['n_samples']:<4} "
                f"Dice={s['dice_mean']:.4f}+/-{s['dice_std']:.4f}  Hit={s['hit_rate']:.4f}"
            )
        if args.visualize:
            logger.info(f"Wrote {len(records)} visualization(s) -> {viz_dir}")


if __name__ == "__main__":
    main()
