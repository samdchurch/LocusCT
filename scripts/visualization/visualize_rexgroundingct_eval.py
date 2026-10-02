#!/usr/bin/env python3
"""
Visualize ReXGroundingCT val predictions written by evaluate_rexgroundingct_val.py.

That script writes one 4D (finding, D, H, W) NIfTI per volume to <eval_dir>/gt and
<eval_dir>/pred, both already in the RAS+ canonical space GrounderDataset trains and
infers in (see its own docstring for how findings are stacked and ordered). This script
re-associates each stacked finding with its manifest entry (sentence + source CT image),
loads that source image through the same canonicalization (load_nifti_canonical) so it
lines up with the GT/pred stacks, and renders one PNG per (volume, finding) --
axial/coronal/sagittal slices centered on the GT mask, with the GT contour in green and
the predicted contour in red, matching visualize.py's style but reading straight from
disk instead of running the model.

Usage
-----
    python visualize_rexgroundingct_eval.py --eval-dir outputs/eval/rexgroundingct_val \
        --data-root /path/to/data
    python visualize_rexgroundingct_eval.py --eval-dir outputs/eval/rexgroundingct_val --n-samples 50
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
from matplotlib.lines import Line2D
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import load_nifti_canonical

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "ReXGroundingCT_val.json"
DEFAULT_DATA_ROOT = Path("/path/to/data")

# Matches evaluate_rexgroundingct_val.py's own convention: mask filenames are
# "<volume_stem>_<finding_idx>.nii.gz"; stem itself may end in digits, so greedily match
# everything before the LAST "_<digits>" as the stem.
STEM_FINDING_RE = re.compile(r"^(?P<stem>.+)_(?P<idx>\d+)$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", type=Path, default=Path("outputs/eval/rexgroundingct_val"),
                         help="Output dir from evaluate_rexgroundingct_val.py (contains gt/, pred/)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                         help="Base dir the manifest's relative image paths resolve against")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Where to write PNGs. Default: <eval-dir>/viz")
    parser.add_argument("--n-samples", type=int, default=100, help="Max (volume, finding) PNGs to write")
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


def _mask_centroid(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask > 0.5)
    if len(coords) == 0:
        return tuple(s // 2 for s in mask.shape)
    d, h, w = coords.mean(axis=0).astype(int)
    d = int(np.clip(d, 0, mask.shape[0] - 1))
    h = int(np.clip(h, 0, mask.shape[1] - 1))
    w = int(np.clip(w, 0, mask.shape[2] - 1))
    return d, h, w


def _dice(gt: np.ndarray, pred: np.ndarray, smooth: float = 1.0) -> float:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    intersection = np.logical_and(gt, pred).sum()
    return float((2.0 * intersection + smooth) / (gt.sum() + pred.sum() + smooth))


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


def main() -> None:
    args = parse_args()

    gt_dir = args.eval_dir / "gt"
    pred_dir = args.eval_dir / "pred"
    output_dir = args.output_dir or (args.eval_dir / "viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    # Group manifest entries by volume stem -> {finding_idx: entry}, so a stacked
    # (finding, ...) NIfTI's array positions can be mapped back to a sentence + image.
    by_stem: dict[str, dict[int, dict]] = defaultdict(dict)
    for entry in manifest:
        if entry.get("mask") is None:
            continue
        stem, idx = parse_stem_and_finding(entry["mask"])
        by_stem[stem][idx] = entry

    # evaluate_rexgroundingct_val.py drops a finding from the stack (without dropping
    # its manifest entry) when its GT file is missing; account for that so the
    # reconstructed index order still matches what's actually in the stack.
    skip_indices_by_stem: dict[str, set[int]] = defaultdict(set)
    skipped_path = args.eval_dir / "skipped.json"
    if skipped_path.exists():
        with open(skipped_path) as f:
            for item in json.load(f):
                if item.get("reason") == "gt_not_found" and "finding_idx" in item:
                    skip_indices_by_stem[item["stem"]].add(item["finding_idx"])

    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    if not gt_files:
        logger.error(f"No .nii.gz files found in {gt_dir}")
        sys.exit(1)

    n_written = 0
    for gt_path in tqdm(gt_files, desc="Volumes"):
        if args.n_samples and n_written >= args.n_samples:
            break

        stem = strip_nii_ext(gt_path.name)
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            logger.warning(f"No matching pred for {stem}, skipping")
            continue
        if stem not in by_stem:
            logger.warning(f"No manifest entries found for {stem}, skipping")
            continue

        gt_4d = np.asanyarray(nib.load(str(gt_path)).dataobj).astype(np.float32)
        pred_4d = np.asanyarray(nib.load(str(pred_path)).dataobj).astype(np.float32)

        idxs = sorted(i for i in by_stem[stem] if i not in skip_indices_by_stem[stem])
        if len(idxs) != gt_4d.shape[0]:
            logger.warning(
                f"{stem}: manifest implies {len(idxs)} finding(s) but the stack has "
                f"{gt_4d.shape[0]}; falling back to positional order (sentences may be wrong)"
            )
            idxs = list(range(gt_4d.shape[0]))

        image_rel = next(iter(by_stem[stem].values()))["image"]
        image_path = args.data_root / image_rel
        if not image_path.exists():
            logger.warning(f"CT image not found for {stem}: {image_path}, skipping")
            continue
        image_dhw = load_nifti_canonical(str(image_path))
        if image_dhw.shape != gt_4d.shape[1:]:
            logger.warning(
                f"{stem}: image shape {image_dhw.shape} doesn't match mask shape "
                f"{gt_4d.shape[1:]}, skipping"
            )
            continue

        for pos, idx in enumerate(idxs):
            if args.n_samples and n_written >= args.n_samples:
                break

            entry = by_stem[stem].get(idx)
            sentence = entry["sentence"] if entry else f"(unmatched finding index {idx})"

            gt_slice = gt_4d[pos]
            pred_slice = pred_4d[pos]

            dice = _dice(gt_slice, pred_slice)
            out_path = output_dir / f"{_safe_filename(f'{stem}_{idx}')}.png"
            _save_figure(image_dhw, gt_slice, pred_slice, sentence, dice, out_path)
            n_written += 1

    logger.info(f"Wrote {n_written} visualization(s) to {output_dir}")


if __name__ == "__main__":
    main()
