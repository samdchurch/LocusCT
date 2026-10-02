#!/usr/bin/env python3
"""
2x5 summary figure: 5 fixed-finding ED cases (top row -- one each of
Aneurysm [splenic artery only], Diverticulitis, Appendicitis, Hernia, and
Hematoma, in that column order; see ED_REQUIRED_FINDINGS) + 5 random
oncology cases (bottom row), each showing the coronal slice (at the GT
mask centroid) of the original raw image with GT mask (green contour) and
predicted mask (red contour) overlaid. The referring sentence renders in
its own panel directly above the image (not overlaid on it, so it never
obscures anatomy), while the Dice score overlays the image's bottom-left
corner in white text with a dark backing box for legibility over variable
anatomy brightness. Oncology's picks (and ties within an ED finding) are
seeded for reproducibility.

Reads already-saved predicted masks from --ed-output-dir/predicted_masks
and --onc-output-dir/predicted_masks (evaluate_ed_official_test.py /
evaluate_onc_official_test.py's --save_masks output, resampled onto each
case's original raw image grid) -- no model, no inference here. Candidates
are restricted to cases that actually have a predicted mask file (and a
resolvable raw image + GT mask) before sampling, so the random pick never
lands on a missing case, and further restricted to a full-volume Dice
above --min-dice (default 0.2, checked by loading and comparing candidates
on the fly until enough qualifying ones are found -- see _pick_ed_cases /
_pick_onc_cases), so the figure doesn't end up illustrating a near-total miss.

These are raw, un-resampled volumes, where in-plane spacing and slice
thickness are usually different (e.g. <1mm vs 3-5mm) -- each coronal slice
is resampled to isotropic (square) pixels using that case's own voxel
spacing (_canonical_axis_info(), read from the NIfTI header -- resolved
from the original DICOM at conversion time, so no need to re-parse DICOM
directly) before anything else, so a plain "equal" aspect displays it
without squishing. The coronal axis itself is also identified per-case
this way (see _canonical_axis_info), rather than assumed, since a handful
of series in this dataset are natively coronal-reconstructed rather than
axial, which otherwise silently produced an axial-looking slice instead.

All 8 selected cases' (now-isotropic) coronal slices are pixel-array-cropped
down to the smallest one's (height, width) -- see _crop_to_size() -- so
every panel displays at the same size. Resampling to isotropic pixels
first is what makes that true: cropping to the same array shape doesn't by
itself guarantee the same *displayed* size if panels still used their own
(different) physical aspect ratios, since imshow would then letterbox each
one differently within its (equal-sized) axes box. The crop window is
centered on each case's own GT mask position within its slice (not just
the slice's geometric center), so shrinking a case with a larger field of
view down to a smaller one's size doesn't risk cropping its segmented
region out of frame.

Also writes a JSON log alongside the figure (same path, ".json" extension --
"test_summary.json" for the default --output) recording exactly which case
went in each panel: id (manifest mask path), finding, sentence, Dice, and
the resolved image/gt/pred file paths, split by cohort.

Usage
-----
    python visualize_test_summary.py --seed 0 --output test_summary.png

Defaults to test_results/test_results/{ed_test,onc_test} (where the actual
runs' output landed); pass --ed-output-dir/--onc-output-dir to point
elsewhere.
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
from matplotlib.lines import Line2D
from scipy.ndimage import zoom

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import load_nifti_canonical

HU_DISPLAY_WINDOW = (-150, 250)
# Fixed finding lineup for the ED row (order = column order), instead of a random
# sample -- Oncology's row is still a random sample of --n cases.
ED_REQUIRED_FINDINGS = ["Aneurysm", "Diverticulitis", "Appendicitis", "Hernia", "Hematoma"]
# Shared between _sentence_panel's wrapping and main()'s sentence-row height calc --
# kept in sync here rather than duplicated as separate literals in each.
SENTENCE_WRAP_WIDTH = 26
SENTENCE_FONTSIZE = 11


def _pred_mask_filename(sample_id: str) -> str:
    """Exactly evaluate_{ed,onc}_official_test.py's _save_masks naming."""
    return f"{sample_id.replace('/', '_')}_pred.nii.gz"


def _canonical_axis_info(path: str) -> dict:
    """
    For load_nifti_canonical()'s output array of `path`: which final axis holds
    each anatomical direction (S-I, A-P, L-R), and that axis's physical spacing
    (mm) -- tracked through the same steps as evaluate_{ed,onc}_official_test.py's
    _canonical_mask_affine().

    nib.as_closest_canonical + transpose(2, 1, 0) (load_nifti_canonical's first two
    steps) reliably place (S-I, A-P, L-R) at pre-heuristic axes (0, 1, 2) -- RAS
    canonicalization is affine-based, not a guess. _normalize_depth_axis's
    subsequent reordering *is* a guess (whichever axis differs in size from the
    other two moves to position 0, on the assumption that's always S-I): for a
    volume whose native reconstruction plane isn't axial -- e.g. a coronally
    reconstructed series, where A-P is the coarse through-plane axis instead of
    S-I -- it moves a *different* anatomical axis to position 0 than usual. This
    mirrors that exact heuristic (same size comparisons) to recover which final
    axis is which, instead of assuming axis 0 is always S-I and axis 1 always A-P.
    """
    canonical = nib.as_closest_canonical(nib.load(path))
    zx, zy, zz = canonical.header.get_zooms()[:3]
    zooms_pre = (zz, zy, zx)  # pre-heuristic (S-I, A-P, L-R) order, per the transpose(2, 1, 0) above

    s0, s1, s2 = canonical.shape[::-1]  # shape in that same pre-heuristic order
    if s1 == s2:
        perm = (0, 1, 2)
    elif s0 == s2:
        perm = (1, 0, 2)  # mirrors _normalize_depth_axis's moveaxis(arr, 1, 0)
    elif s0 == s1:
        perm = (2, 0, 1)  # mirrors _normalize_depth_axis's moveaxis(arr, 2, 0)
    else:
        perm = (0, 1, 2)

    return {
        "si_axis": perm.index(0),
        "ap_axis": perm.index(1),
        "lr_axis": perm.index(2),
        "zooms": tuple(zooms_pre[p] for p in perm),  # final-axis-order spacings
    }


def _dice(gt: np.ndarray, pred: np.ndarray, smooth: float = 1.0) -> float:
    gt = gt > 0.5
    pred = pred > 0.5
    intersection = np.logical_and(gt, pred).sum()
    return float((2.0 * intersection + smooth) / (gt.sum() + pred.sum() + smooth))


def _mask_centroid(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask > 0.5)
    if len(coords) == 0:
        return tuple(s // 2 for s in mask.shape)
    d, h, w = coords.mean(axis=0).astype(int)
    return int(d), int(h), int(w)


def _collect_candidates(manifest_path: Path, raw_image_dir: Path, gt_mask_dir: Path, pred_mask_dir: Path) -> list[dict]:
    with open(manifest_path) as f:
        samples = json.load(f)

    candidates = []
    for s in samples:
        if not s.get("sentence"):
            continue
        image_path = raw_image_dir / s["image"]
        gt_path = gt_mask_dir / s["mask"]
        pred_path = pred_mask_dir / _pred_mask_filename(s["mask"])
        if image_path.exists() and gt_path.exists() and pred_path.exists():
            candidates.append({
                "id": s["mask"], "image": image_path, "gt": gt_path, "pred": pred_path,
                "sentence": s["sentence"], "finding": s.get("finding"),
            })
    return candidates


def _pick_ed_cases(candidates: list[dict], rng: random.Random, min_dice: float) -> tuple[list[dict], list[dict]]:
    """One candidate per finding in ED_REQUIRED_FINDINGS, in that order, restricted to
    candidates whose actual (full-volume) Dice exceeds min_dice. The Aneurysm slot is
    further restricted to splenic artery aneurysms -- the manifest also has
    aortic/iliac aneurysm cases under the same "Aneurysm" finding, which don't
    qualify. Dice isn't known until a candidate is loaded, so candidates within each
    finding are tried in a random (rng-seeded, so reproducible) order and loaded one
    at a time until one clears the threshold, rather than loading the whole pool
    upfront. Returns (picks, loaded) so a qualifying candidate isn't loaded twice."""
    picks, loaded = [], []
    for finding in ED_REQUIRED_FINDINGS:
        pool = [c for c in candidates if c["finding"] == finding]
        if finding == "Aneurysm":
            pool = [c for c in pool if "splenic" in c["sentence"].lower()]
        if not pool:
            qualifier = " (splenic)" if finding == "Aneurysm" else ""
            raise ValueError(f"No ED candidate with a saved prediction for finding={finding!r}{qualifier}")
        rng.shuffle(pool)
        for c in pool:
            data = _load_case(c)
            if data["dice"] > min_dice:
                picks.append(c)
                loaded.append(data)
                break
        else:
            raise ValueError(f"No ED candidate for finding={finding!r} has Dice > {min_dice} "
                              f"(checked {len(pool)} candidate(s) with a saved prediction)")
    return picks, loaded


def _pick_onc_cases(
    candidates: list[dict], rng: random.Random, n: int, min_dice: float
) -> tuple[list[dict], list[dict]]:
    """Random sample of up to n candidates whose actual (full-volume) Dice exceeds
    min_dice -- same on-the-fly load-and-check reasoning as _pick_ed_cases, trying
    candidates in a random (rng-seeded) order without replacement until n qualifying
    cases are found or the pool is exhausted."""
    pool = list(candidates)
    rng.shuffle(pool)
    picks, loaded = [], []
    for c in pool:
        if len(picks) >= n:
            break
        data = _load_case(c)
        if data["dice"] > min_dice:
            picks.append(c)
            loaded.append(data)
    if len(picks) < n:
        raise ValueError(f"Only found {len(picks)}/{n} oncology candidate(s) with Dice > {min_dice} "
                          f"(checked all {len(pool)} candidate(s) with a saved prediction)")
    return picks, loaded


def _load_case(case: dict) -> dict:
    """Loads image/GT/pred, extracts the coronal slice at the GT mask centroid
    (rotated to display orientation, HU-clipped), and computes the case's own
    Dice (full volume). The slice is then resampled to isotropic pixels (see
    below) and not cropped yet -- main() crops every selected case to a
    shared size afterward, via _crop_to_size."""
    lo, hi = HU_DISPLAY_WINDOW
    image = load_nifti_canonical(str(case["image"]))
    gt = load_nifti_canonical(str(case["gt"]))
    pred = load_nifti_canonical(str(case["pred"]))

    # Which final array axis is A-P (fix it -- that's what makes the slice
    # coronal), and which are S-I / L-R (keep both -- those are the slice's own
    # two display dimensions). See _canonical_axis_info for why this can't just
    # assume axis 0 is S-I and axis 1 is A-P.
    axes = _canonical_axis_info(str(case["image"]))
    coronal_axis, si_axis, lr_axis = axes["ap_axis"], axes["si_axis"], axes["lr_axis"]
    h = _mask_centroid(gt)[coronal_axis]

    def _coronal_slice(vol: np.ndarray) -> np.ndarray:
        idx = [slice(None)] * 3
        idx[coronal_axis] = h
        sl = vol[tuple(idx)]
        # Basic indexing keeps the two non-fixed axes in their original relative
        # order; reorient to (S-I, L-R) specifically if that came out reversed.
        remaining = [ax for ax in range(3) if ax != coronal_axis]
        return sl if remaining[0] == si_axis else sl.T

    img_sl = np.rot90(np.clip(_coronal_slice(image), lo, hi), 2)
    gt_sl = np.rot90(_coronal_slice(gt), 2)
    pred_sl = np.rot90(_coronal_slice(pred), 2)

    # Coronal slice is (D, W) with different physical spacing per axis (raw,
    # un-resampled volumes: slice thickness != in-plane spacing). Passing that
    # as imshow's `aspect` merely letterboxes the *display* -- once every case
    # is cropped to the same array shape (see main()), panels with very
    # different aspect ratios would still end up rendered at very different
    # sizes within their (equal-sized) axes box. Resampling the array itself
    # to isotropic (square) pixels here instead means "same array shape" and
    # "same displayed size" are the same thing, and a plain "equal" aspect is
    # then correct for every panel.
    spacing_d, spacing_w = axes["zooms"][si_axis], axes["zooms"][lr_axis]
    aspect = spacing_d / spacing_w
    img_sl = zoom(img_sl, (aspect, 1), order=1)
    gt_sl = zoom(gt_sl, (aspect, 1), order=0)
    pred_sl = zoom(pred_sl, (aspect, 1), order=0)
    print(f"  {case['image'].name}: coronal_axis={coronal_axis} spacing D={spacing_d:.3f}mm W={spacing_w:.3f}mm "
          f"-> aspect={aspect:.3f}  resampled slice shape={img_sl.shape}")

    return {"img_sl": img_sl, "gt_sl": gt_sl, "pred_sl": pred_sl, "dice": _dice(gt, pred)}


def _crop_to_size(data: dict, target_h: int, target_w: int) -> None:
    """In-place center crop of img_sl/gt_sl/pred_sl to (target_h, target_w) --
    centered on the GT mask's own position within the slice (falling back to
    the slice's geometric center if empty), clamped to stay in-bounds, so
    shrinking every panel to the smallest selected case's size can't crop the
    segmented region out of view for a case whose mask sits off-center."""
    h, w = data["img_sl"].shape
    coords = np.argwhere(data["gt_sl"] > 0.5)
    cy, cx = coords.mean(axis=0) if len(coords) > 0 else (h / 2, w / 2)
    top = int(np.clip(round(cy - target_h / 2), 0, h - target_h))
    left = int(np.clip(round(cx - target_w / 2), 0, w - target_w))
    for key in ("img_sl", "gt_sl", "pred_sl"):
        data[key] = data[key][top:top + target_h, left:left + target_w]


def _sentence_panel(ax, sentence: str) -> None:
    """Renders the referring sentence in its own axes directly above the
    matching image panel -- not overlaid on it, so it never obscures the
    anatomy."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    wrapped = textwrap.fill(sentence, width=SENTENCE_WRAP_WIDTH)
    ax.text(0.5, 0.0, wrapped, transform=ax.transAxes, ha="center", va="bottom",
            color="black", fontsize=SENTENCE_FONTSIZE)


def _render_panel(ax, data: dict) -> None:
    """Renders one already-loaded (resampled to isotropic pixels and cropped)
    case: image + GT/pred contours, with the Dice score overlaid in white
    text over the bottom-left corner (a dark backing box keeps it legible
    regardless of how bright that area's anatomy happens to be)."""
    # aspect="equal" can't fill a cell whose box aspect doesn't match the
    # (isotropic, but not necessarily square) cropped slice -- matplotlib
    # shrinks the image to fit and, by default, centers the leftover space.
    # Anchoring to the top instead pushes any leftover padding to the bottom
    # of the cell, where it isn't visible between the image and the text above it.
    ax.set_anchor("N")
    ax.imshow(data["img_sl"], cmap="gray", aspect="equal", origin="upper")
    if data["gt_sl"].any():
        ax.contour(data["gt_sl"], levels=[0.5], colors=["#00e676"], linewidths=1.5)
    if data["pred_sl"].any():
        ax.contour(data["pred_sl"], levels=[0.5], colors=["#ff1744"], linewidths=1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    dice_part = r"$\mathbf{Dice:\ " + f"{data['dice']:.3f}" + "}$"
    ax.text(0.03, 0.03, dice_part, transform=ax.transAxes, ha="left", va="bottom",
            color="white", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5, edgecolor="none"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ed-manifest", default="official_splits/ed_official_test_data.json")
    parser.add_argument("--onc-manifest", default="official_splits/onc_official_test_data.json")
    parser.add_argument("--raw-image-dir", default="/path/to/data/inhouse_abdominal_ct/nifti")
    parser.add_argument("--ed-gt-mask-dir", default="/path/to/data/ED_EXAMPLES_DATASET/NIFTI_DATA")
    parser.add_argument("--onc-gt-mask-dir", default="/path/to/data/inhouse_abdominal_ct/ALL_LABELS")
    parser.add_argument("--ed-output-dir", default="/path/to/repo/test_results/test_results/ed_test",
                         help="Dir passed as --output's parent to evaluate_ed_official_test.py")
    parser.add_argument("--onc-output-dir", default="/path/to/repo/test_results/test_results/onc_test",
                         help="Dir passed as --output's parent to evaluate_onc_official_test.py")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-dice", type=float, default=0.2,
                         help="Only consider candidates whose (full-volume) Dice exceeds this")
    parser.add_argument("--output", default="outputs/figures/test_summary.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_image_dir = Path(args.raw_image_dir)
    rng = random.Random(args.seed)

    ed_candidates = _collect_candidates(
        Path(args.ed_manifest), raw_image_dir, Path(args.ed_gt_mask_dir), Path(args.ed_output_dir) / "predicted_masks"
    )
    onc_candidates = _collect_candidates(
        Path(args.onc_manifest), raw_image_dir, Path(args.onc_gt_mask_dir), Path(args.onc_output_dir) / "predicted_masks"
    )
    print(f"ED candidates with a saved prediction: {len(ed_candidates)}")
    print(f"Oncology candidates with a saved prediction: {len(onc_candidates)}")

    print("Selecting and loading cases...")
    ed_picks, ed_loaded = _pick_ed_cases(ed_candidates, rng, args.min_dice)
    onc_picks, onc_loaded = _pick_onc_cases(onc_candidates, rng, args.n, args.min_dice)

    all_loaded = ed_loaded + onc_loaded
    min_h = min(d["img_sl"].shape[0] for d in all_loaded)
    min_w = min(d["img_sl"].shape[1] for d in all_loaded)
    print(f"Cropping all panels to {min_h}x{min_w} (smallest selected case)")
    for d in all_loaded:
        _crop_to_size(d, min_h, min_w)

    # Size the image rows to the cropped panels' own (isotropic-pixel) aspect ratio,
    # rather than a guessed height_ratio -- imshow's aspect="equal" doesn't stretch to
    # fill its axes box, so a box taller than the image's native aspect just left empty
    # space beneath it. Column width is kept at the previous fixed 4in; only the height
    # derivation changed.
    col_width_in = 4.0
    image_h_in = col_width_in * (min_h / min_w)

    # Sentence row height is sized to the longest wrapped sentence among the selected
    # cases, not a fixed guess -- a long referring expression would otherwise overflow
    # its axes and visually bleed into the image row above/below it.
    all_sentences = [p["sentence"] for p in ed_picks + onc_picks]
    max_lines = max(len(textwrap.fill(s, width=SENTENCE_WRAP_WIDTH).split("\n")) for s in all_sentences)
    sentence_h_in = max_lines * SENTENCE_FONTSIZE * 1.4 / 72 + 0.2  # 1.4x line spacing + a little padding

    fig = plt.figure(figsize=(col_width_in * args.n, 2 * sentence_h_in + 2 * image_h_in), facecolor="white")
    # 4 grid rows = 2 cohorts x (sentence sub-row, image sub-row).
    gs = fig.add_gridspec(
        4, args.n, height_ratios=[sentence_h_in, image_h_in, sentence_h_in, image_h_in], hspace=0.0, wspace=0.2
    )

    cohorts = [
        ("Emergency\nDepartment", ed_picks, ed_loaded),
        ("Oncology", onc_picks, onc_loaded),
    ]

    image_axes = [None, None]
    for cohort_idx, (label, picks, loaded) in enumerate(cohorts):
        sentence_row = cohort_idx * 2
        image_row = sentence_row + 1

        row_axes = []
        for col in range(args.n):
            ax_sent = fig.add_subplot(gs[sentence_row, col])
            ax_img = fig.add_subplot(gs[image_row, col])
            row_axes.append(ax_img)
            if col < len(picks):
                _sentence_panel(ax_sent, picks[col]["sentence"])
                _render_panel(ax_img, loaded[col])
            else:
                ax_sent.axis("off")
                ax_img.axis("off")
        image_axes[cohort_idx] = row_axes
        row_axes[0].set_ylabel(label, color="black", fontsize=26, fontweight="bold")

    legend = [
        Line2D([0], [0], color="#00e676", linewidth=4, label="GT"),
        Line2D([0], [0], color="#ff1744", linewidth=4, label="Pred"),
    ]
    image_axes[-1][-1].legend(handles=legend, loc="lower right", fontsize=16, prop={"weight": "bold"},
                               handlelength=2.5, handleheight=1.5, borderpad=0.8, labelspacing=0.6,
                               framealpha=0.9, facecolor="white", labelcolor="black", edgecolor="black")

    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.4, facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.output}")

    log_path = Path(args.output).with_suffix(".json")
    log = {
        "seed": args.seed,
        "ed": [
            {"id": pick["id"], "finding": pick["finding"], "sentence": pick["sentence"], "dice": data["dice"],
             "image": str(pick["image"]), "gt": str(pick["gt"]), "pred": str(pick["pred"])}
            for pick, data in zip(ed_picks, ed_loaded)
        ],
        "onc": [
            {"id": pick["id"], "finding": pick["finding"], "sentence": pick["sentence"], "dice": data["dice"],
             "image": str(pick["image"]), "gt": str(pick["gt"]), "pred": str(pick["pred"])}
            for pick, data in zip(onc_picks, onc_loaded)
        ],
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
