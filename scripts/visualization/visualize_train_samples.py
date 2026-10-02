#!/usr/bin/env python3
"""
Quick QC dump: N random samples from the Grounder training manifest, each
showing its CT image with the GT segmentation mask (green contour, axial/
coronal/sagittal), its referring sentence, and the original GSPS annotation
(points/measurement overlaid on the matching slice of that same image, per
--gsps-dir) -- no model, no predictions, just a look at the raw training data.

Resolves image/mask paths the same way GrounderDataset does (manifest's
relative "image"/"mask" fields joined onto --image-dir/--mask-dir), but
skips the dataset class entirely -- no tokenizer/model needed just to look
at images, so this has no transformers/torch dependency. Defaults to the
original, un-resampled nifti/ALL_LABELS directories (not the nifti_resampled/
labels_resampled ones GrounderDataset actually trains on) since this is for
eyeballing source data, not verifying model input.

Any sample whose image or mask file can't be loaded is skipped with a
warning rather than aborting the whole run, since data-root layout has been
a moving target in this environment -- partial success beats an
all-or-nothing crash. The GSPS panel is best-effort per sample: masks named
"mask_box_*" (merged from two source annotations -- see
update_box_annotations.py) have no single original annotation to show, and
any sample whose per-accession GSPS export is missing or has no matching
series/slice/annotation just gets a "No GSPS annotation" placeholder panel
rather than being skipped outright.

Usage
-----
    python visualize_train_samples.py --n 20 --seed 0 --output-dir outputs/viz/train_samples
"""
import argparse
import json
import random
import re
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

# "{accession}/mask_{series}_{index}_{anno_idx}.nii.gz" -- see update_box_annotations.py's
# mask_rel_path. Doesn't match "mask_box_..." names, which merge two source annotations and
# so have no single original annotation to look up.
MASK_NAME_RE = re.compile(r"^mask_(\d+)_(\d+)_(\d+)$")


def _mask_centroid(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask > 0.5)
    if len(coords) == 0:
        return tuple(s // 2 for s in mask.shape)
    d, h, w = coords.mean(axis=0).astype(int)
    return int(d), int(h), int(w)


def _parse_mask_name(mask_rel: str) -> tuple[str, int, int, int] | None:
    """Returns (accession, series, index, anno_idx), or None for box masks / unrecognized names."""
    parts = Path(mask_rel).parts
    if len(parts) < 2:
        return None
    accession = parts[0]
    stem = parts[-1]
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    m = MASK_NAME_RE.match(stem)
    if not m:
        return None
    series, index, anno_idx = (int(g) for g in m.groups())
    return accession, series, index, anno_idx


def _find_gsps_annotation(
    gsps_dir: Path, accession: str, series: int, index: int, anno_idx: int
) -> tuple[dict, int] | None:
    """Returns (annotation, nifti_idx), or None if the export/series/slice/annotation isn't found.
    Schema: [{"series", "slices": [{"index", "annotations": [...], "nifti_idx"}]}] -- same per-accession
    GSPS export format."""
    gsps_path = gsps_dir / f"{accession}.json"
    if not gsps_path.exists():
        return None
    with open(gsps_path) as f:
        data = json.load(f)
    for series_entry in data:
        if series_entry.get("series") != series:
            continue
        for slice_entry in series_entry.get("slices", []):
            if slice_entry.get("index") != index:
                continue
            annotations = slice_entry.get("annotations", [])
            if anno_idx < len(annotations):
                return annotations[anno_idx], slice_entry.get("nifti_idx")
    return None


def _canonical_axis_info(image_path: str) -> dict:
    """Which final array axis (of load_nifti_canonical's output) holds each anatomical
    direction (S-I, A-P, L-R), and each axis's physical spacing (mm) -- ported from
    visualize_test_summary.py's _canonical_axis_info (see its docstring for the full
    reasoning). Needed because axis 0 isn't always S-I: load_nifti_canonical's
    _normalize_depth_axis picks whichever axis differs in size from the other two,
    which for most series is S-I (axial acquisitions have thick axial slices) but for
    a handful of series in this dataset that are natively coronal-reconstructed
    (thick coronal slices instead) is A-P -- assuming axis 0 is always S-I would then
    silently slice/aspect-ratio the "Axial" panel as if it were coronal, and vice
    versa.
    """
    canonical = nib.as_closest_canonical(nib.load(image_path))
    zx, zy, zz = canonical.header.get_zooms()[:3]
    zooms_pre = (zz, zy, zx)  # pre-heuristic (S-I, A-P, L-R) order, per load_nifti_canonical's transpose(2, 1, 0)
    s0, s1, s2 = canonical.shape[::-1]  # shape in that same pre-heuristic order
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
        "zooms": tuple(zooms_pre[p] for p in perm),  # final-axis-order spacings
    }


def _plane_slice(vol: np.ndarray, fixed_axis: int, fixed_idx: int, row_axis: int) -> np.ndarray:
    """2D slice of vol with `fixed_axis` pinned to fixed_idx, oriented so the row_axis
    (of the two remaining axes) ends up first -- basic indexing keeps the two
    non-fixed axes in their original relative order, which needs a transpose to
    guarantee a consistent layout regardless of which axis got fixed."""
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


def _raw_slice_axis(shape: tuple[int, int, int]) -> int:
    """Which axis of a *raw* (on-disk, pre-canonicalization) volume is the through-plane/
    slice axis that nifti_idx indexes into -- whichever axis differs in size from the
    other two (e.g. the axis that isn't 512, for a 512x512xN axial series), rather than
    always assuming it's the last axis. dcm2niix writes the slice axis wherever the
    source series' own plane puts it, so a coronal- or sagittal-native series can have
    it at position 0 or 1 instead of 2."""
    s0, s1, s2 = shape
    if s1 == s2:
        return 0
    elif s0 == s2:
        return 1
    elif s0 == s1:
        return 2
    return 2  # no clean pair -- keep the prior (always-last-axis) behavior


def _raw_axis_to_canonical(ornt: np.ndarray, raw_axis: int, axis_info: dict) -> int:
    """Which canonical (D, H, W) array position -- axis_info's si_axis/ap_axis/lr_axis --
    a *raw* (on-disk) axis corresponds to, per its RAS role (ornt[raw_axis, 0]: 0=R, 1=A,
    2=S) in the same orientation info nib.as_closest_canonical itself uses."""
    return [axis_info["lr_axis"], axis_info["ap_axis"], axis_info["si_axis"]][int(ornt[raw_axis, 0])]


def _raw_value_to_canonical(ornt: np.ndarray, raw_axis: int, size: int, value: float) -> float:
    """A raw-space index/coordinate along raw_axis, converted to the canonical (RAS+)
    array's coordinate along the same physical axis -- reversed (size - 1 - value) if
    that raw axis runs in the negative direction relative to canonical (ornt[raw_axis, 1]
    < 0), unchanged otherwise."""
    return (size - 1 - value) if ornt[raw_axis, 1] < 0 else value


def _plot_gsps_panel(ax: plt.Axes, gsps: dict | None) -> None:
    """gsps is None (no annotation found) or {"slice", "xs", "ys", "aspect"} -- a slice of
    the same canonically-oriented `image` array the Axial/Coronal/Sagittal panels use
    (see main()'s mapping of the GSPS export's raw-space index/points into that space),
    already rotated/clipped the same way, plus its matching aspect. Point-count branches
    handle each GSPS annotation shape by its number of points."""
    if gsps is None:
        ax.text(0.5, 0.5, "No GSPS\nannotation", color="gray", ha="center", va="center",
                fontsize=10, transform=ax.transAxes)
    else:
        ax.imshow(gsps["slice"], cmap="gray", aspect=gsps["aspect"], origin="upper")
        xs, ys = gsps["xs"], gsps["ys"]
        if len(xs) == 2:
            ax.plot(xs, ys, color="#ff00ff", linewidth=0.75, marker="o", markersize=1.5)
        elif len(xs) == 5:
            ax.plot(xs[0:2], ys[0:2], color="#ff00ff", linewidth=0.75, marker="o", markersize=1.5)
            ax.plot(xs[3:5], ys[3:5], color="#ff00ff", linewidth=0.75, marker="o", markersize=1.5)
        else:
            ax.plot(xs, ys, color="#ff00ff", linewidth=0.75, marker="o", markersize=1.5)
    ax.set_title("GSPS Annotation", color="white", fontsize=11, pad=4)
    ax.set_facecolor("black")
    ax.axis("off")


def _save_figure(
    image: np.ndarray,
    mask: np.ndarray,
    sentence: str,
    sample_id: str,
    out_path: Path,
    axis_info: dict,
    gsps: dict | None = None,
) -> None:
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

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="black")
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
    _plot_gsps_panel(axes[3], gsps)

    fig.suptitle(f"{sample_id}\n" + textwrap.fill(sentence, width=90), color="white", fontsize=9, y=1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="official_splits/all_data_train.json")
    parser.add_argument("--image-dir", default="/path/to/data/inhouse_abdominal_ct/nifti")
    parser.add_argument("--mask-dir", default="/path/to/data/inhouse_abdominal_ct/ALL_LABELS")
    parser.add_argument("--gsps-dir", default="/path/to/data/inhouse_abdominal_ct/deid_gsps")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/viz/train_samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.manifest) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples from {args.manifest}")

    rng = random.Random(args.seed)
    picks = rng.sample(samples, min(args.n, len(samples)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gsps_dir = Path(args.gsps_dir)
    lo, hi = HU_DISPLAY_WINDOW

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

        gsps = None
        parsed = _parse_mask_name(sample["mask"])
        if parsed is not None:
            accession, series, index, anno_idx = parsed
            found = _find_gsps_annotation(gsps_dir, accession, series, index, anno_idx)
            if found is not None:
                annotation, nifti_idx = found
                try:
                    # nifti_idx and the annotation's points are in the *raw* (on-disk,
                    # pre-canonicalization) volume's own axis order. Rather than separately
                    # loading and re-orienting that raw volume for display, map its slice
                    # axis and point coordinates into the already-loaded canonical `image`
                    # array's (D, H, W) space (using the affine's orientation info, same as
                    # nib.as_closest_canonical uses internally), then slice/rotate/aspect
                    # that array exactly the way _save_figure's Axial/Coronal/Sagittal views
                    # do -- guaranteed to match their (already-correct) orientation, since
                    # it's the literal same array and code path, just at the GSPS's own
                    # slice index rather than the mask centroid's.
                    raw_img = nib.load(str(image_path))
                    ornt = nib.io_orientation(raw_img.affine)
                    slice_axis = _raw_slice_axis(raw_img.shape)
                    x_axis, y_axis = (ax for ax in range(3) if ax != slice_axis)

                    fixed_axis = _raw_axis_to_canonical(ornt, slice_axis, axis_info)
                    canonical_idx = int(round(
                        _raw_value_to_canonical(ornt, slice_axis, raw_img.shape[slice_axis], nifti_idx)
                    ))
                    x_canon_axis = _raw_axis_to_canonical(ornt, x_axis, axis_info)
                    xs = [_raw_value_to_canonical(ornt, x_axis, raw_img.shape[x_axis], x)
                          for x in annotation["points"][0::2]]
                    ys = [_raw_value_to_canonical(ornt, y_axis, raw_img.shape[y_axis], y)
                          for y in annotation["points"][1::2]]

                    si_axis, ap_axis, lr_axis = axis_info["si_axis"], axis_info["ap_axis"], axis_info["lr_axis"]
                    if fixed_axis == si_axis:
                        row_axis, col_axis = ap_axis, lr_axis
                    elif fixed_axis == ap_axis:
                        row_axis, col_axis = si_axis, lr_axis
                    else:
                        row_axis, col_axis = si_axis, ap_axis
                    row_vals, col_vals = (xs, ys) if x_canon_axis == row_axis else (ys, xs)

                    img_sl = _plane_slice(image, fixed_axis, canonical_idx, row_axis)
                    img_sl = np.rot90(np.clip(img_sl, lo, hi), 2)
                    n_rows, n_cols = img_sl.shape
                    plot_xs = [(n_cols - 1) - c for c in col_vals]
                    # Confirmed against real samples: the row/y compensation for rot90(k=2) is
                    # needed for axial GSPS slices but must be skipped for coronal/sagittal ones.
                    plot_ys = [(n_rows - 1) - r for r in row_vals] if fixed_axis == si_axis else list(row_vals)
                    zooms = axis_info["zooms"]
                    aspect = zooms[row_axis] / zooms[col_axis] if zooms[col_axis] else 1.0
                    gsps = {"slice": img_sl, "xs": plot_xs, "ys": plot_ys, "aspect": aspect}

                    plane = {si_axis: "axial", ap_axis: "coronal", lr_axis: "sagittal"}[fixed_axis]
                    print(f"  GSPS debug: plane={plane} raw_shape={raw_img.shape} affine_diag={np.diag(raw_img.affine)[:3]}")
                    print(f"    slice_axis={slice_axis} x_axis={x_axis} y_axis={y_axis} "
                          f"fixed_axis={fixed_axis} row_axis={row_axis} col_axis={col_axis}")
                    print(f"    nifti_idx={nifti_idx} -> canonical_idx={canonical_idx}  n_rows={n_rows} n_cols={n_cols}")
                    print(f"    raw points={list(zip(annotation['points'][0::2], annotation['points'][1::2]))}")
                    print(f"    xs(canon)={xs} ys(canon)={ys}  row_vals={row_vals} col_vals={col_vals}")
                    print(f"    plot_xs={plot_xs} plot_ys={plot_ys}")
                except (IndexError, KeyError, OSError) as e:
                    print(f"  GSPS annotation found but couldn't render: {e}")

        out_path = out_dir / f"{_safe_filename(sample['mask'])}.png"
        _save_figure(
            image, mask, sample.get("sentence", sample["mask"]), sample["mask"], out_path, axis_info, gsps=gsps
        )
        print(f"[{i + 1}/{len(picks)}] wrote {out_path}")
        n_ok += 1

    print(f"Done. {n_ok}/{len(picks)} samples visualized -> {out_dir}")


if __name__ == "__main__":
    main()
