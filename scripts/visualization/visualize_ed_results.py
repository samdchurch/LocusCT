#!/usr/bin/env python3
"""
Multi-panel GT/prediction figure for the ED official test set, built from
already-saved predicted masks (e.g. --pred-mask-dir
/path/to/data/temp/ed_test_test/predicted_masks, evaluate_ed_official_test.py's
--save_masks output) -- no model, no checkpoint, no inference here.

Those predicted masks are resampled onto each case's original raw image
grid (see evaluate_ed_official_test.py's _save_masks/_canonical_mask_affine),
not the model's 352x352x180 working resolution -- so this script's image
and GT mask are read from the same raw grid too:
  image:    --raw-image-dir (default .../inhouse_abdominal_ct/nifti), the
            pre-resample_and_crop.py NIfTI files
  GT mask:  --gt-mask-dir (default .../ED_EXAMPLES_DATASET/NIFTI_DATA),
            the un-resampled curated ED masks (ED_TEST_SET_resampled under
            inhouse_abdominal_ct is the resampled-to-match-training copy; this
            uses the original source instead, which lives elsewhere)
  pred mask: --pred-mask-dir, named "{mask_id with '/' -> '_'}_pred.nii.gz"
            -- exactly evaluate_ed_official_test.py's _save_masks naming.

Writes one PNG per case (axial/coronal/sagittal at the GT mask centroid,
green GT contour, red prediction contour, Dice + sentence caption) to
--output-dir/<category>/, matching visualize.py's style. Cases whose pred
mask file is missing are skipped with a warning.

Usage
-----
    python visualize_ed_results.py \\
        --pred-mask-dir /path/to/data/temp/ed_test_test/predicted_masks \\
        --output-dir outputs/viz/ed_results
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import load_nifti_canonical

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "ed_official_test_data.json"
HU_DISPLAY_WINDOW = (-150, 250)


def category_of(sample_id: str) -> str:
    return sample_id.split("/")[0]


def _pred_mask_filename(sample_id: str) -> str:
    """Exactly evaluate_ed_official_test.py's _save_masks naming."""
    return f"{sample_id.replace('/', '_')}_pred.nii.gz"


def _mask_centroid(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask > 0.5)
    if len(coords) == 0:
        return tuple(s // 2 for s in mask.shape)
    d, h, w = coords.mean(axis=0).astype(int)
    return int(d), int(h), int(w)


def _dice(gt: np.ndarray, pred: np.ndarray, smooth: float = 1.0) -> float:
    gt = gt > 0.5
    pred = pred > 0.5
    intersection = np.logical_and(gt, pred).sum()
    return float((2.0 * intersection + smooth) / (gt.sum() + pred.sum() + smooth))


def _safe_filename(sample_id: str) -> str:
    name = sample_id.replace("/", "_").replace("\\", "_")
    for ext in (".nii.gz", ".nii", ".gz"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name[-150:] if len(name) > 150 else name


def _save_figure(image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray,
                  sentence: str, dice: float, out_path: Path) -> None:
    lo, hi = HU_DISPLAY_WINDOW
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
            img_sl = np.rot90(np.clip(img_sl, lo, hi), 2)
            gt_sl = np.rot90(gt_sl, 2)
            pred_sl = np.rot90(pred_sl, 2)
            ax.imshow(img_sl, cmap="gray", vmin=lo, vmax=hi, aspect="equal", origin="upper")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-image-dir", default="/path/to/data/inhouse_abdominal_ct/nifti")
    parser.add_argument("--gt-mask-dir", default="/path/to/data/ED_EXAMPLES_DATASET/NIFTI_DATA")
    parser.add_argument("--pred-mask-dir", required=True,
                         help="e.g. /path/to/data/temp/ed_test_test/predicted_masks")
    parser.add_argument("--output-dir", default="outputs/viz/ed_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.manifest) as f:
        samples = json.load(f)
    samples = [s for s in samples if s.get("sentence")]

    raw_image_dir = Path(args.raw_image_dir)
    gt_mask_dir = Path(args.gt_mask_dir)
    pred_mask_dir = Path(args.pred_mask_dir)
    out_dir = Path(args.output_dir)

    n_ok = 0
    for i, sample in enumerate(samples):
        sample_id = sample["mask"]
        pred_path = pred_mask_dir / _pred_mask_filename(sample_id)
        if not pred_path.exists():
            print(f"[{i + 1}/{len(samples)}] SKIP {sample_id}: no predicted mask at {pred_path}")
            continue

        image_path = raw_image_dir / sample["image"]
        gt_path = gt_mask_dir / sample_id
        missing = []
        if not image_path.exists():
            missing.append(f"image={image_path}")
        if not gt_path.exists():
            missing.append(f"GT mask={gt_path}")
        if missing:
            print(f"[{i + 1}/{len(samples)}] SKIP {sample_id}: missing {', '.join(missing)}")
            continue

        image = load_nifti_canonical(str(image_path))
        gt = load_nifti_canonical(str(gt_path))
        pred = load_nifti_canonical(str(pred_path))
        if not (image.shape == gt.shape == pred.shape):
            print(f"[{i + 1}/{len(samples)}] SKIP {sample_id}: shape mismatch "
                  f"image={image.shape} gt={gt.shape} pred={pred.shape}")
            continue

        dice = _dice(gt, pred)
        cat_dir = out_dir / category_of(sample_id)
        cat_dir.mkdir(parents=True, exist_ok=True)
        _save_figure(image, gt, pred, sample["sentence"], dice, cat_dir / f"{_safe_filename(sample_id)}.png")
        print(f"[{i + 1}/{len(samples)}] wrote {cat_dir / f'{_safe_filename(sample_id)}.png'}  dice={dice:.3f}")
        n_ok += 1

    print(f"Done. {n_ok}/{len(samples)} case(s) visualized -> {out_dir}")


if __name__ == "__main__":
    main()
