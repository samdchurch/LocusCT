#!/usr/bin/env python3
"""
Score the external VoxTell checkpoint on the ReXGroundingCT val split
(official_splits/ReXGroundingCT_val.json), reporting Dice and hit rate
(dice >= 0.1) overall and per region. Writes one GT/pred overlay PNG per
case if --visualize is passed, matching evaluate_voxtell_ed.py's style.

Like evaluate_voxtell_ed.py, this runs VoxTell on native-resolution NIfTI
volumes rather than our own model's resampled 352x352x180 grid. The
manifest's image/mask fields point at ".../ReXGroundingCT/resampled/...",
our own resampling output (resample_rexgroundingct.py); the corresponding
native files live at the same relative path with "resampled/" swapped for
"original/" -- native images: ReXGroundingCT/original/images/<split>_fixed/...;
native segmentations: ReXGroundingCT/original/segmentations/<stem>.nii.gz,
ONE 4D file per volume stacking every finding, unlike the per-finding 3D
files our own resampling step split them into.

Orientation caveat (NOT empirically validated the way the ED script's
native pairing was -- see its kidney/liver/spleen sanity check): resample_
rexgroundingct.py's own docstring warns the native segmentation's embedded
affine "is not trustworthy for spatial resampling". So rather than
reorienting each finding's slice through NibabelIOWithReorient via its own
(possibly-wrong) affine, this borrows the paired image's affine for the
segmentation before canonicalizing with nibabel's as_closest_canonical --
the same technique resample_rexgroundingct.py itself uses (matching image
to mask by array shape, not trusting the mask's own affine). Which axis of
the native 4D file holds the per-finding dimension is also unconfirmed, so
_finding_axis() detects it by elimination against the image's shape rather
than assuming a fixed convention. Treat a first run's skipped-sample count
as the signal for whether these hold up -- a high skip rate means one of
these assumptions needs revisiting on real data.

Samples are grouped by source volume (stem parsed from the manifest's mask
filename, "<stem>_<finding_idx>.nii.gz") so each scan is only encoded once
by VoxTell regardless of how many findings it has.

Usage
-----
    python evaluate_voxtell_rexgroundingct.py
    python evaluate_voxtell_rexgroundingct.py --output outputs/eval/voxtell_rex/results.json
"""

import argparse
import json
import logging
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
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

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "ReXGroundingCT_val.json"
DEFAULT_DATA_ROOT = "/path/to/data/public_datasets"
DEFAULT_MODEL_DIR = "/path/to/repo/voxtell/voxtell_v1.1"
DEFAULT_TEXT_ENCODER = "/path/to/data/models/Qwen3-Embedding-4B"

# Mask filenames are "<volume_stem>_<finding_idx>.nii.gz"; stem itself may end in
# digits, so greedily match everything before the LAST "_<digits>" as the stem
# (same convention as evaluate_rexgroundingct_val.py's STEM_FINDING_RE).
STEM_FINDING_RE = re.compile(r"^(?P<stem>.+)_(?P<idx>\d+)$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                         help="Base dir the native 'ReXGroundingCT/original/images|segmentations/...' paths resolve against")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--text-encoding-model", default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--output", default="outputs/eval/voxtell_rex/results.json")
    parser.add_argument("--hit-threshold", type=float, default=0.1,
                         help="Dice value at/above which a sample counts as a 'hit'")
    parser.add_argument("--visualize", dest="visualize", action="store_true", default=False,
                         help="Write per-case GT/pred overlay PNGs (off by default)")
    parser.add_argument("--viz-output-dir", default=None,
                         help="Where to write visualization PNGs. Default: <output's parent>/viz")
    return parser.parse_args()


def strip_nii_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return name


def parse_stem_and_finding(mask_rel_path: str) -> tuple[str, int]:
    stem_with_idx = strip_nii_ext(Path(mask_rel_path).name)
    m = STEM_FINDING_RE.match(stem_with_idx)
    if not m:
        raise ValueError(f"Can't parse volume stem/finding index from mask path: {mask_rel_path}")
    return m["stem"], int(m["idx"])


def native_image_path(data_root: str, resampled_image_rel: str) -> str:
    """'ReXGroundingCT/resampled/images/...' -> 'ReXGroundingCT/original/images/...'"""
    native_rel = resampled_image_rel.replace("ReXGroundingCT/resampled/", "ReXGroundingCT/original/", 1)
    return f"{data_root}/{native_rel}"


def native_seg_path(data_root: str, stem: str) -> str:
    return f"{data_root}/ReXGroundingCT/original/segmentations/{stem}.nii.gz"


def _finding_axis(seg_shape: tuple, img_shape: tuple) -> int:
    """Native segmentation is 4D (findings + 3 spatial axes matching img_shape),
    but which axis holds the findings isn't confirmed -- detect it by elimination
    rather than assuming a fixed convention."""
    if seg_shape[1:] == img_shape:
        return 0
    if seg_shape[:3] == img_shape:
        return 3
    raise ValueError(f"Can't find a findings axis: seg_shape={seg_shape} img_shape={img_shape}")


def region_of(sample: dict) -> str:
    return sample.get("region") or "UNKNOWN"


def _mask_centroid(mask: np.ndarray) -> tuple[int, int, int]:
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
    of a long run can't wipe out everything computed so far."""
    by_region = defaultdict(list)
    for r in records:
        by_region[r["region"]].append(r)

    summary = {
        "model_dir": model_dir,
        "hit_threshold": hit_threshold,
        "overall": summarize(records),
        "by_region": {reg: summarize(recs) for reg, recs in sorted(by_region.items())},
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
        logger.warning(f"Dropping {n_dropped}/{len(samples)} sample(s) with no sentence")

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

    by_stem: dict[str, list[dict]] = defaultdict(list)
    for s in kept:
        stem, finding_idx = parse_stem_and_finding(s["mask"])
        by_stem[stem].append({**s, "_finding_idx": finding_idx})

    records: list[dict] = []
    skipped: list[dict] = []

    try:
        for group_idx, (stem, group) in enumerate(
            tqdm(sorted(by_stem.items()), desc="Evaluating ReXGroundingCT val (VoxTell)")
        ):
            group = sorted(group, key=lambda s: s["_finding_idx"])
            img_path = native_image_path(args.data_root, group[0]["image"])
            seg_path = native_seg_path(args.data_root, stem)

            try:
                img, _ = reader.read_images([img_path])
                raw_img_nib = nib.load(img_path)
                sentences = [s["sentence"] for s in group]
                pred = predictor.predict_single_image(img, sentences)  # (n, X, Y, Z)
                image_np = img[0]

                raw_seg_nib = nib.load(seg_path)
                raw_seg_data = raw_seg_nib.get_fdata()
                finding_axis = _finding_axis(raw_seg_data.shape, raw_img_nib.shape)
            except Exception as e:
                logger.warning(f"SKIP volume {stem}: {e}")
                skipped.extend({"stem": stem, "mask": s["mask"], "reason": str(e)} for s in group)
                continue

            for i, s in enumerate(group):
                mask_rel = s["mask"]
                try:
                    finding_idx = s["_finding_idx"]
                    finding_arr = (
                        raw_seg_data[finding_idx] if finding_axis == 0
                        else raw_seg_data[..., finding_idx]
                    )
                    # Borrow the image's own affine rather than the segmentation
                    # file's -- resample_rexgroundingct.py's docstring warns the
                    # latter isn't trustworthy here, and array indices already
                    # correspond 1:1 with the paired image's own grid.
                    finding_nib = nib.Nifti1Image(finding_arr.astype(np.uint8), affine=raw_img_nib.affine)
                    canonical_finding = nib.as_closest_canonical(finding_nib)
                    gt = (canonical_finding.get_fdata() > 0.5).astype(np.float32)
                    pr = pred[i].astype(np.float32)
                    if gt.shape != pr.shape:
                        raise ValueError(f"image/mask shape mismatch: pred={pr.shape} gt={gt.shape}")

                    dice = dice_score(
                        torch.from_numpy(pr[None]), torch.from_numpy(gt[None]),
                        threshold=0.5, smooth=1.0, from_logits=False,
                    )[0].item()

                    reg = region_of(s)
                    records.append({
                        "id": mask_rel,
                        "region": reg,
                        "dice": dice,
                        "hit": bool(dice >= args.hit_threshold),
                    })

                    if args.visualize:
                        reg_dir = viz_dir / reg
                        reg_dir.mkdir(parents=True, exist_ok=True)
                        _save_figure(
                            image_np, gt, pr, s["sentence"], dice,
                            reg_dir / f"{_safe_filename(mask_rel)}.png",
                        )
                except Exception as e:
                    logger.warning(f"SKIP mask {mask_rel}: {e}")
                    skipped.append({"stem": stem, "mask": mask_rel, "reason": str(e)})
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
        logger.info("  By region:")
        for reg, s in summary["by_region"].items():
            logger.info(
                f"    {reg:<15} n={s['n_samples']:<4} "
                f"Dice={s['dice_mean']:.4f}+/-{s['dice_std']:.4f}  Hit={s['hit_rate']:.4f}"
            )
        if args.visualize:
            logger.info(f"Wrote {len(records)} visualization(s) -> {viz_dir}")


if __name__ == "__main__":
    main()
