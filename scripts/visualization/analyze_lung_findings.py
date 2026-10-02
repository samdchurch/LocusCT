#!/usr/bin/env python3
"""
Counts how many of the Grounder training manifest's samples describe a lung
finding, breaks those down by finding type, and renders a QC dump (axial/
coronal/sagittal, GT mask contour, sentence) for a random sample of them --
using the lung HU window (data.dataset.MULTI_WINDOWS[0], the same range
GrounderDataset's multi_window mode uses for its "lung" channel) rather than
visualize_train_samples.py's soft-tissue window, since a soft-tissue window
mostly renders lung parenchyma as a featureless dark blob.

"Lung finding" is determined by keyword search over each sample's "sentence"
text (see LUNG_KEYWORDS) -- there's no organ/anatomy field to check
directly. The manifest's own "region" field is a broad body-region label
("Abdomen"/"Chest"/...), not organ-specific: plenty of "Chest"-region
entries are mediastinal/cardiac/skeletal, not lung parenchyma, so it can't
be used as a lung filter on its own. The keyword list was checked against
this manifest during development: it captures ~10k samples with the finding
types you'd expect (Nodule, Mass, Groundglass Opacity, Consolidation,
Pulmonary Embolism, ...), with a small (~1-2%) false-positive rate from
"apical" occasionally matching non-lung anatomy (e.g. "apical prostate
mass", dental "periapical lucency") -- acceptable for a QC/exploration
tool, not specifically filtered out here.

Any sample whose image or mask file can't be loaded is skipped with a
warning rather than aborting the whole run (same reasoning as
visualize_train_samples.py).

Usage
-----
    python analyze_lung_findings.py --n 20 --seed 0 --output-dir outputs/viz/lung_findings
    python analyze_lung_findings.py --n 0  # visualize every lung finding, not just a sample
"""
import argparse
import json
import random
import re
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import MULTI_WINDOWS, load_nifti_canonical

LUNG_WINDOW = MULTI_WINDOWS[0]  # (-1000.0, 300.0) -- GrounderDataset's multi_window "lung" channel range

# Case-insensitive substring keywords identifying a lung-parenchyma finding from sentence
# text -- see module docstring for validation notes / known limitations.
LUNG_KEYWORDS = [
    "lung", "pulmonary", "upper lobe", "lower lobe", "middle lobe", "lingula",
    "groundglass", "ground-glass", "ground glass", "atelectasis", "bronchovascular",
    "bronchiect", "bronchiol", "costophrenic", "apical", "endobronchial",
]
LUNG_KEYWORD_RE = re.compile("|".join(re.escape(k) for k in LUNG_KEYWORDS), re.IGNORECASE)


def is_lung_finding(sentence: str | None) -> bool:
    return bool(LUNG_KEYWORD_RE.search(sentence or ""))


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


def _safe_filename(sample_id: str) -> str:
    name = sample_id.replace("/", "_").replace("\\", "_")
    for ext in (".nii.gz", ".nii", ".gz"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name[-150:] if len(name) > 150 else name


def _save_figure(
    image: np.ndarray, mask: np.ndarray, sentence: str, sample_id: str, out_path: Path, axis_info: dict
) -> None:
    lo, hi = LUNG_WINDOW
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
            ax.contour(mask_sl, levels=[0.5], colors=["#00e676"], linewidths=1.5)
        ax.set_title(label, color="white", fontsize=11, pad=4)
        ax.set_facecolor("black")
        ax.axis("off")

    fig.suptitle(f"{sample_id}\n" + textwrap.fill(sentence, width=90), color="white", fontsize=9, y=1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="official_splits/all_data_train.json")
    parser.add_argument("--image-dir", default="/path/to/data/inhouse_abdominal_ct/nifti")
    parser.add_argument("--mask-dir", default="/path/to/data/inhouse_abdominal_ct/ALL_LABELS")
    parser.add_argument("--n", type=int, default=20, help="Number of lung-finding samples to visualize (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/viz/lung_findings")
    parser.add_argument("--counts-output", default=None,
                         help="Where to write the finding-type counts as JSON "
                              "(default: <output-dir>/lung_finding_counts.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.manifest) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples from {args.manifest}")

    lung_samples = [s for s in samples if is_lung_finding(s.get("sentence"))]
    print(f"Lung findings: {len(lung_samples)}/{len(samples)} ({100 * len(lung_samples) / len(samples):.1f}%)")

    finding_counts = Counter(s.get("finding") or "(none)" for s in lung_samples)
    print(f"\n{len(finding_counts)} distinct finding type(s) among lung findings:")
    for finding, count in finding_counts.most_common():
        print(f"  {count:6d}  {finding}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts_path = Path(args.counts_output) if args.counts_output else out_dir / "lung_finding_counts.json"
    with open(counts_path, "w") as f:
        json.dump(
            {
                "n_total_samples": len(samples),
                "n_lung_findings": len(lung_samples),
                "finding_counts": dict(finding_counts.most_common()),
            },
            f, indent=2,
        )
    print(f"\nWrote finding counts -> {counts_path}")

    rng = random.Random(args.seed)
    picks = lung_samples if args.n == 0 else rng.sample(lung_samples, min(args.n, len(lung_samples)))

    n_ok = 0
    for i, sample in enumerate(picks):
        image_path = Path(args.image_dir) / sample["image"]
        mask_path = Path(args.mask_dir) / sample["mask"]
        try:
            image = load_nifti_canonical(str(image_path))
            mask = load_nifti_canonical(str(mask_path))
            axis_info = _canonical_axis_info(str(image_path))
        except (FileNotFoundError, OSError) as e:
            print(f"[{i + 1}/{len(picks)}] SKIP {sample['mask']}: {e}")
            continue

        out_path = out_dir / f"{_safe_filename(sample['mask'])}.png"
        _save_figure(image, mask, sample.get("sentence", sample["mask"]), sample["mask"], out_path, axis_info)
        print(f"[{i + 1}/{len(picks)}] wrote {out_path}")
        n_ok += 1

    print(f"\nDone. {n_ok}/{len(picks)} lung-finding samples visualized -> {out_dir}")


if __name__ == "__main__":
    main()
