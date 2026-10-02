#!/usr/bin/env python3
"""
QC dump for predict_merlin.py's output: N random (study, finding) predictions
from --predictions (its predictions.json), each showing the CT image with
the predicted mask (red contour, axial/coronal/sagittal) and the finding
sentence -- no ground truth here, this is Merlin data, just a look at what
the model predicted.

Loads the image via load_nifti_canonical (same RAS+ canonicalization used
everywhere else in this repo), but loads the predicted mask with a plain
nib.load(...).get_fdata() -- NOT load_nifti_canonical -- since
predict_merlin.py already saved it in that same canonical (D, H, W) array
orientation with a placeholder affine; re-running load_nifti_canonical on it
would incorrectly re-transpose it (see predict_merlin.py's module docstring).

Any sample whose image or mask file can't be loaded is skipped with a
warning rather than aborting the whole run.

Usage
-----
    python visualize_merlin_predictions.py --predictions merlin_predictions/predictions.json --n 20
    python visualize_merlin_predictions.py --predictions merlin_predictions/predictions.json --study-id AC423ccbe --n 0
"""
import argparse
import json
import random
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import load_nifti_canonical

HU_DISPLAY_WINDOW = (-150, 250)
PRED_COLOR = "#ff1744"  # matches make_results_figure.py / visualize_ed_results.py's pred-mask color


def _mask_centroid(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask > 0.5)
    if len(coords) == 0:
        return tuple(s // 2 for s in mask.shape)
    d, h, w = coords.mean(axis=0).astype(int)
    return int(d), int(h), int(w)


def _canonical_axis_info(image_path: str) -> dict:
    """Ported from visualize_train_samples.py -- see its docstring for the full reasoning
    (axis 0 of load_nifti_canonical's output isn't always S-I, e.g. for coronally-native
    series)."""
    canonical = nib.as_closest_canonical(nib.load(image_path))
    zx, zy, zz = canonical.header.get_zooms()[:3]
    zooms_pre = (zz, zy, zx)
    s0, s1, s2 = canonical.shape[::-1]
    if s1 == s2:
        perm = (0, 1, 2)
    elif s0 == s2:
        perm = (1, 0, 2)
    elif s0 == s1:
        perm = (2, 0, 1)
    else:
        perm = (0, 1, 2)
    return {
        "si_axis": perm.index(0),
        "ap_axis": perm.index(1),
        "lr_axis": perm.index(2),
        "zooms": tuple(zooms_pre[p] for p in perm),
    }


def _plane_slice(vol: np.ndarray, fixed_axis: int, fixed_idx: int, row_axis: int) -> np.ndarray:
    idx: list = [slice(None)] * 3
    idx[fixed_axis] = fixed_idx
    sl = vol[tuple(idx)]
    remaining = [ax for ax in range(3) if ax != fixed_axis]
    return sl if remaining[0] == row_axis else sl.T


def _safe_filename(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_")
    return name[-150:] if len(name) > 150 else name


def _save_figure(image: np.ndarray, mask: np.ndarray, title: str, out_path: Path, axis_info: dict) -> None:
    lo, hi = HU_DISPLAY_WINDOW
    si_axis, ap_axis, lr_axis = axis_info["si_axis"], axis_info["ap_axis"], axis_info["lr_axis"]
    zooms = axis_info["zooms"]
    centroid = _mask_centroid(mask)

    def _view(fixed_axis: int, row_axis: int, col_axis: int) -> tuple[np.ndarray, np.ndarray, float]:
        img_sl = _plane_slice(image, fixed_axis, centroid[fixed_axis], row_axis)
        mask_sl = _plane_slice(mask, fixed_axis, centroid[fixed_axis], row_axis)
        aspect = zooms[row_axis] / zooms[col_axis] if zooms[col_axis] else 1.0
        return img_sl, mask_sl, aspect

    views = [
        ("Axial", *_view(si_axis, ap_axis, lr_axis)),
        ("Coronal", *_view(ap_axis, si_axis, lr_axis)),
        ("Sagittal", *_view(lr_axis, si_axis, ap_axis)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="black")
    fig.patch.set_facecolor("black")
    for ax, (label, img_sl, mask_sl, asp) in zip(axes, views):
        img_sl = np.rot90(np.clip(img_sl, lo, hi), 2)
        mask_sl = np.rot90(mask_sl, 2)
        ax.imshow(img_sl, cmap="gray", aspect=asp, origin="upper")
        if mask_sl.any():
            ax.contour(mask_sl, levels=[0.5], colors=[PRED_COLOR], linewidths=1.5)
        ax.set_title(label, color="white", fontsize=11, pad=4)
        ax.set_facecolor("black")
        ax.axis("off")

    fig.suptitle(textwrap.fill(title, width=100), color="white", fontsize=9, y=1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="predictions.json written by predict_merlin.py")
    parser.add_argument("--study-id", default=None, help="Only visualize predictions for this study_id")
    parser.add_argument("--n", type=int, default=20, help="Number of samples to visualize (0 = all matching)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-empty", action="store_true", help="Skip predictions with an empty (all-zero) mask")
    parser.add_argument("--output-dir", default="outputs/viz/merlin_predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.predictions) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} prediction(s) from {args.predictions}")

    if args.study_id:
        records = [r for r in records if r["study_id"] == args.study_id]
    if args.skip_empty:
        records = [r for r in records if r.get("voxel_count", 0) > 0]

    rng = random.Random(args.seed)
    picks = records if args.n == 0 else rng.sample(records, min(args.n, len(records)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for i, rec in enumerate(picks):
        image_path = rec["image_path"]
        mask_path = rec["mask_path"]
        try:
            image = load_nifti_canonical(image_path)
            mask = nib.load(mask_path).get_fdata()
            axis_info = _canonical_axis_info(image_path)
        except (FileNotFoundError, OSError) as e:
            print(f"[{i + 1}/{len(picks)}] SKIP {rec['study_id']}_finding{rec['finding_idx']}: {e}")
            continue

        title = (f"{rec['study_id']}  finding {rec['finding_idx']}  "
                 f"({rec.get('organ', '')}, {rec.get('laterality', '')})\n{rec['sentence']}")
        out_name = _safe_filename(f"{rec['study_id']}_finding{rec['finding_idx']}") + ".png"
        out_path = out_dir / out_name
        _save_figure(image, mask, title, out_path, axis_info)
        print(f"[{i + 1}/{len(picks)}] wrote {out_path}  (voxels={rec.get('voxel_count', '?')})")
        n_ok += 1

    print(f"Done. {n_ok}/{len(picks)} samples visualized -> {out_dir}")


if __name__ == "__main__":
    main()
