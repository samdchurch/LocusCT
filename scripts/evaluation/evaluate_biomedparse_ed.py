#!/usr/bin/env python3
"""
Score the external BiomedParse-v2 model on the official ED held-out test set
(official_splits/ed_official_test_data.json), reporting Dice and hit rate
(dice >= 0.1) overall and per finding category. Pass --visualize to also
write one GT/pred overlay PNG per case (axial/coronal/sagittal, green GT
contour, red predicted contour) to <output's parent>/viz/<category>/,
matching evaluate_voxtell_ed.py's style.

Like evaluate_voxtell_ed.py, this runs on native-resolution NIfTI volumes
paired against native (pre-resample) masks -- not the 352x352x180 resampled
grid evaluate_ed_official_test.py uses -- so these Dice numbers aren't a
strictly apples-to-apples comparison against that script's numbers.

Model: BiomedParse v2 (biomedparse-v2/BiomedParse), loaded the same way as
biomedparse-v2/segmentation_example.py -- hydra-instantiated from
configs/model/biomedparse_3D.yaml and loaded from the local
biomedparse_v2.ckpt (not load_model.py's HF-hub 2D SEEM-style API, which is
a different, older model).

Preprocessing: raw HU volumes are windowed and rescaled to [0, 255] before
process_input(), per the BiomedParse-v2 README's "Recommended Preprocessing"
section (the packaged demo *.npz files are already in this range, which is
why segmentation_example.py itself doesn't show this step). Default window
is the "soft tissue" window (W:400, L:40), appropriate for abdominal CT.

Each finding gets its own single-prompt forward pass (not batched with other
findings on the same scan). BiomedParse v2's demo pattern batches same-scan
findings into one multi-prompt call and merges them via merge_multiclass_masks
(an argmax across findings) -- fast, but it makes overlapping findings on the
same scan mutually exclusive, which would distort per-finding Dice. Calling
per-finding avoids that at the cost of re-running the image backbone once per
finding instead of once per scan.

Usage
-----
    python evaluate_biomedparse_ed.py
    python evaluate_biomedparse_ed.py --output outputs/eval/biomedparse_ed/results.json
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
import torch.nn.functional as F
from matplotlib.lines import Line2D
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from tqdm import tqdm

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "ed_official_test_data.json"
DEFAULT_IMAGE_DIR = "/path/to/data/inhouse_abdominal_ct/nifti"
DEFAULT_MASK_DIR = "/path/to/data/ED_EXAMPLES_DATASET/NIFTI_DATA"
DEFAULT_BIOMEDPARSE_REPO = Path(__file__).resolve().parent / "biomedparse-v2" / "BiomedParse"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "biomedparse-v2" / "model" / "biomedparse_v2.ckpt"

# (window width, window level) in HU, per BiomedParse-v2 README's "Recommended Preprocessing"
CT_WINDOWS = {
    "soft_tissue": (400, 40),
    "lung": (1500, -160),
    "brain": (80, 40),
    "bone": (1800, 400),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Populated by load_model() once biomedparse_repo is known (needs sys.path set up first).
process_input = None
process_output = None
postprocess = None
merge_multiclass_masks = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR,
                         help="Base dir the manifest's relative 'image' paths resolve against")
    parser.add_argument("--mask-dir", default=DEFAULT_MASK_DIR,
                         help="Base dir the manifest's relative 'mask' paths resolve against")
    parser.add_argument("--biomedparse-repo", type=Path, default=DEFAULT_BIOMEDPARSE_REPO,
                         help="Path to the cloned BiomedParse source repo (has utils.py, inference.py, configs/)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--clip-tokenizer-dir", default=None,
                         help="Local directory holding a pre-downloaded 'openai/clip-vit-base-patch32' "
                              "tokenizer (config.json/vocab.json/merges.txt/tokenizer_config.json -- no "
                              "model weights needed, see seem_language_encoder.yaml's LOAD_PRETRAINED: "
                              "false). BiomedParse-v2's own hydra config hardcodes that HF Hub ID for its "
                              "internal CLIP tokenizer; on a cluster with no internet egress that fetch "
                              "hard-fails (OSError, not a graceful fallback) rather than using a cache. "
                              "Default: leave the config's HF Hub ID as-is (works if you have internet "
                              "access or it's already properly HF-cached).")
    parser.add_argument("--ct-window", choices=sorted(CT_WINDOWS), default="soft_tissue",
                         help="HU windowing applied before rescaling to [0, 255]")
    parser.add_argument("--target-size", type=int, default=512, help="In-plane size fed to the model")
    parser.add_argument("--slice-batch-size", type=int, default=4)
    parser.add_argument("--output", default="outputs/eval/biomedparse_ed/results.json")
    parser.add_argument("--hit-threshold", type=float, default=0.1,
                         help="Dice value at/above which a sample counts as a 'hit'")
    parser.add_argument("--visualize", dest="visualize", action="store_true", default=False,
                         help="Write per-case GT/pred overlay PNGs (off by default)")
    parser.add_argument("--viz-output-dir", default=None,
                         help="Where to write visualization PNGs. Default: <output's parent>/viz")
    return parser.parse_args()


def load_model(
    biomedparse_repo: Path, checkpoint: Path, device: torch.device, clip_tokenizer_dir: str | None = None,
):
    """Load BiomedParse v2 the same way biomedparse-v2/segmentation_example.py does.

    biomedparse-v2's utils.py/inference.py use bare top-level imports (not a proper
    installable package), so the repo must be on sys.path before they can be imported --
    hence the deferred `global` binding here rather than module-level imports.

    clip_tokenizer_dir: overrides configs/model/sem_seg_head/predictor/language_encoder/
    seem_language_encoder.yaml's tokenizer.pretrained_model_name_or_path (hardcoded there to
    the HF Hub ID 'openai/clip-vit-base-patch32') to a local directory instead -- see
    --clip-tokenizer-dir's help text for why.
    """
    global process_input, process_output, postprocess, merge_multiclass_masks

    biomedparse_repo = biomedparse_repo.resolve()  # hydra.initialize_config_dir requires an absolute path
    sys.path.insert(0, str(biomedparse_repo))
    from utils import process_input as _process_input, process_output as _process_output
    from inference import postprocess as _postprocess, merge_multiclass_masks as _merge_multiclass_masks
    process_input, process_output = _process_input, _process_output
    postprocess, merge_multiclass_masks = _postprocess, _merge_multiclass_masks

    import hydra
    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(
        config_dir=str(biomedparse_repo / "configs" / "model"), job_name="ed_eval", version_base=None
    )
    overrides = []
    if clip_tokenizer_dir:
        overrides.append(
            f"sem_seg_head.predictor.language_encoder.tokenizer.pretrained_model_name_or_path={clip_tokenizer_dir}"
        )
    cfg = compose(config_name="biomedparse_3D", overrides=overrides)
    model = hydra.utils.instantiate(cfg, _convert_="object")
    model.load_pretrained(str(checkpoint))
    return model.to(device).eval()


def window_ct(volume_hu: np.ndarray, width: float, level: float) -> np.ndarray:
    """HU volume -> float32 in [0, 255], per the BiomedParse-v2 README's preprocessing recipe.

    Kept as float (not cast to uint8) because process_input()'s bicubic resize requires a
    floating dtype; the model-facing example scripts cast to int only *after* process_input.
    """
    lo, hi = level - width / 2, level + width / 2
    v = np.clip(volume_hu, lo, hi).astype(np.float32)
    return (v - lo) / (hi - lo) * 255.0


def load_and_preprocess_image(image_hu: np.ndarray, window: tuple[float, float], target_size: int, device):
    image_u8 = window_ct(image_hu, *window)
    imgs, pad_width, padded_size, valid_axis = process_input(image_u8, target_size)
    return imgs.to(device).int(), pad_width, padded_size, valid_axis


def biomedparse_predict_one(
    model, imgs: torch.Tensor, pad_width, padded_size, valid_axis, sentence: str,
    target_size: int, slice_batch_size: int,
) -> np.ndarray:
    """Single-prompt forward pass. Returns a (X, Y, Z) float32 binary mask matching image_hu.shape."""
    input_tensor = {"image": imgs.unsqueeze(0), "text": [sentence]}
    with torch.no_grad():
        output = model(input_tensor, mode="eval", slice_batch_size=slice_batch_size)

    mask_preds = output["predictions"]["pred_gmasks"]
    mask_preds = F.interpolate(
        mask_preds, size=(target_size, target_size), mode="bicubic", align_corners=False, antialias=True
    )
    mask_preds = postprocess(mask_preds, output["predictions"]["object_existence"])
    class_mask = merge_multiclass_masks(mask_preds, [1])  # single prompt -> {0=bg, 1=fg}
    pred = process_output(class_mask, pad_width, padded_size, valid_axis)
    return (pred == 1).astype(np.float32)


def dice_score_np(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5, smooth: float = 1.0) -> float:
    pred = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)
    intersection = (pred * target).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def category_of(sample: dict) -> str:
    """Groups by the manifest's own "finding" field -- LocusBench's mask paths
    are prefixed with "masks/" (unlike the old official manifest's bare
    "CATEGORY/accession/..."), so path parsing no longer recovers the
    category."""
    return sample.get("finding") or "Unknown"


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
    checkpoint: str,
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
        "checkpoint": checkpoint,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = NibabelIOWithReorient()
    logger.info(f"Loading BiomedParse from {args.biomedparse_repo} (checkpoint={args.checkpoint})")
    model = load_model(args.biomedparse_repo, args.checkpoint, device, args.clip_tokenizer_dir)
    window = CT_WINDOWS[args.ct_window]

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
            tqdm(sorted(by_image.items()), desc="Evaluating ED official test set (BiomedParse)")
        ):
            try:
                img, _ = reader.read_images([f"{args.image_dir}/{image_rel}"])
                image_np = img[0]
                imgs, pad_width, padded_size, valid_axis = load_and_preprocess_image(
                    image_np, window, args.target_size, device
                )
            except Exception as e:
                logger.warning(f"SKIP image group {image_rel}: {e}")
                skipped.extend({"image": image_rel, "mask": s["mask"], "reason": str(e)} for s in group)
                continue

            for s in group:
                mask_rel = s["mask"]
                try:
                    pr = biomedparse_predict_one(
                        model, imgs, pad_width, padded_size, valid_axis, s["sentence"],
                        args.target_size, args.slice_batch_size,
                    )
                    seg, _ = reader.read_seg(f"{args.mask_dir}/{mask_rel}")
                    gt = (seg[0] > 0.5).astype(np.float32)
                    if gt.shape != pr.shape:
                        raise ValueError(f"image/mask shape mismatch: pred={pr.shape} gt={gt.shape} "
                                          f"(mask likely drawn on a different reconstruction/series than "
                                          f"the manifest's 'image' field points to)")

                    dice = dice_score_np(pr, gt, threshold=0.5, smooth=1.0)

                    cat = category_of(s)
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
                _write_summary(output_path, str(args.checkpoint), args.hit_threshold, records, skipped)
            torch.cuda.empty_cache()
    finally:
        summary = _write_summary(output_path, str(args.checkpoint), args.hit_threshold, records, skipped)

        overall = summary["overall"]
        logger.info(
            f"Results -> {output_path}\n"
            f"  Overall ({overall['n_samples']} samples, {len(skipped)} skipped):\n"
            f"    Dice: {overall['dice_mean']:.4f} +/- {overall['dice_std']:.4f}   "
            f"Hit: {overall['hit_rate']:.4f}"
        )
        logger.info("  By category:")
        for cat, s in summary["by_category"].items():
            logger.info(
                f"    {cat:<15} n={s['n_samples']:<4} "
                f"Dice={s['dice_mean']:.4f}+/-{s['dice_std']:.4f}  Hit={s['hit_rate']:.4f}"
            )
        if args.visualize:
            logger.info(f"Wrote {len(records)} visualization(s) -> {viz_dir}")


if __name__ == "__main__":
    main()
