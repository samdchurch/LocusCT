#!/usr/bin/env python3
"""
Score the published SegVol_v1.pth checkpoint (github.com/BAAI-DCAI/SegVol) on the
official ED held-out test set (official_splits/ed_official_test_data.json), reporting
Dice and hit rate (dice >= 0.1) overall and per finding category. Pass --visualize to
also write one GT/pred overlay PNG per case (axial/coronal/sagittal, green GT contour,
red predicted contour) to <output's parent>/viz/<category>/, matching the other
baselines' eval scripts' style.

Like the other baselines' eval scripts, this runs on native-resolution NIfTI volumes
and isn't a strictly apples-to-apples comparison against this repo's own
352x352x180-resampled-grid eval scripts.

Model: SegVol, run via SegVol's own real preprocessing (data_process.demo_data_process
.process_ct_gt) and its own real "zoom-in-zoom-out" inference loop
(inference_demo.zoom_in_zoom_out, imported directly from the repo's own demo script)
-- both entirely unmodified. process_ct_gt already returns image and GT mask aligned
on the identical processed grid (RAS orientation, foreground-mean-based intensity
normalization, min-max scaling, spatial pad, foreground crop), so unlike BiomedParse
and SAT's own eval scripts there's no need to separately re-derive that grid for
scoring -- zoom_in_zoom_out's own returned logits are already resized back onto it.
Dice is SegVol's own dice_score() (imported from inference_demo.py), not a
reimplementation -- it happens to use the exact same (2*intersection+1)/(sum+sum+1)
formula as the other three baselines' eval scripts.

Two decisions made explicit here, confirmed with the user before writing this script
(see conversation): the other two prompt-related choices SegVol's own demo makes by
default are NOT reproduced --

1. SegVol is run **text-prompt only** (use_box_prompt=False, use_point_prompt=False),
   not inference_demo.py's own default combination of text+box. That default's box
   prompt is generated from the *ground-truth mask itself*
   (data_process/demo_data_process.py-processed labels -> generate_box() in
   inference_demo.py's zoom_in_zoom_out()), which would leak GT spatial location into
   the prediction -- not a fair comparison against VoxTell/BiomedParse/SAT, which all
   get text only. Text-only is a real, SegVol-supported prompt mode (the paper
   describes point/box/text as independently usable), just not the demo's default
   combination. The "zoom-in-zoom-out" mechanism itself stays on and is unaffected --
   its ROI-detection step (logits2roi_coor) works off the model's own first-pass
   prediction, not the GT.

2. SegVol's text encoder (network/model.py::TextEncoder.organ2tokens) wraps the query
   in a fixed template ('A computerized tomography of a {}.') and was trained on short
   category names (its own demo uses "liver", "kidney", etc.), not free-text sentences
   -- and its CLIP text tower is capped at 77 tokens with no truncation set on the
   tokenizer call, so passing one of this dataset's referring expressions (up to 500+
   characters) directly would crash with a position-embedding index error. Per the
   user's direction, the actual referring-expression sentence is still what gets sent
   (not swapped for a shorter category label) -- truncate_sentence_for_clip() below
   pre-truncates it to fit the template's token budget using the real CLIP tokenizer
   (token-level, not a naive character cut), so as much of the real sentence survives
   as fits. This doesn't touch SegVol's own TextEncoder code.

One more accommodation, not a judgment call -- a defensive correctness fix:
process_ct_gt (data_process/demo_data_process.py) extracts each category's mask via
exact integer equality (gt_voxel_ndarray == cls, cls=1..N for a length-N category
list), correct for SegVol's own training data (clean multi-class integer label maps)
but not something safe to assume about this repo's own mask files sight-unseen -- every
other eval script in this repo instead thresholds GT with `> 0.5`. binarize_mask_to_tempfile()
below writes a clean {0,1} int16 copy of each mask (thresholded the same way) to a temp
file before handing it to process_ct_gt, so a mismatched on-disk convention (e.g. 255
for foreground) can't silently turn every GT mask empty. Doesn't touch process_ct_gt
itself.

Usage
-----
    python evaluate_segvol_ed.py --checkpoint /path/to/SegVol_v1.pth
"""

import argparse
import json
import logging
import sys
import tempfile
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
from transformers import AutoTokenizer

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "ed_official_test_data.json"
DEFAULT_IMAGE_DIR = "/path/to/data/inhouse_abdominal_ct/nifti"
DEFAULT_MASK_DIR = "/path/to/data/ED_EXAMPLES_DATASET/NIFTI_DATA"
DEFAULT_SEGVOL_REPO = Path(__file__).resolve().parents[2] / "segvol" / "SegVol"

# CLIPTextConfig()'s default max_position_embeddings (network/model.py::TextEncoder
# builds a bare CLIPTextConfig(), not one loaded from the tokenizer's own config.json).
CLIP_MAX_POSITION_EMBEDDINGS = 77
TEMPLATE = "A computerized tomography of a {}."

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
    parser.add_argument("--segvol-repo", type=Path, default=DEFAULT_SEGVOL_REPO,
                         help="Path to the cloned SegVol source repo (has network/, segment_anything_volumetric/, data_process/)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="SegVol_v1.pth")
    parser.add_argument("--clip-ckpt", type=Path, default=None,
                         help="Path to the CLIP tokenizer dir. Default: <segvol-repo>/config/clip")
    parser.add_argument("--spatial-size", type=int, nargs=3, default=[32, 256, 256])
    parser.add_argument("--patch-size", type=int, nargs=3, default=[4, 16, 16])
    parser.add_argument("--infer-overlap", type=float, default=0.5, help="sliding window inference overlap")
    parser.add_argument("--output", default="outputs/eval/segvol_ed/results.json")
    parser.add_argument("--hit-threshold", type=float, default=0.1,
                         help="Dice value at/above which a sample counts as a 'hit'")
    parser.add_argument("--visualize", dest="visualize", action="store_true", default=False,
                         help="Write per-case GT/pred overlay PNGs (off by default)")
    parser.add_argument("--viz-output-dir", default=None,
                         help="Where to write visualization PNGs. Default: <output's parent>/viz")
    args = parser.parse_args()
    args.spatial_size = tuple(args.spatial_size)
    args.patch_size = tuple(args.patch_size)
    if args.clip_ckpt is None:
        args.clip_ckpt = args.segvol_repo / "config" / "clip"
    # Attributes zoom_in_zoom_out()/build_sam_vit_3d() read directly off `args`.
    args.test_mode = True
    args.use_zoom_in = True
    args.use_text_prompt = True
    args.use_box_prompt = False
    args.use_point_prompt = False
    return args


# ---------------------------------------------------------------------------
# CLIP truncation (see module docstring, decision 2)
# ---------------------------------------------------------------------------

def build_clip_tokenizer(clip_ckpt: Path):
    return AutoTokenizer.from_pretrained(str(clip_ckpt))


def max_sentence_tokens(tokenizer) -> int:
    template_ids = tokenizer(TEMPLATE.format(""), add_special_tokens=True)["input_ids"]
    return max(1, CLIP_MAX_POSITION_EMBEDDINGS - len(template_ids) - 2)  # small safety margin


def truncate_sentence_for_clip(tokenizer, sentence: str, budget: int) -> str:
    ids = tokenizer(sentence, add_special_tokens=False, truncation=True, max_length=budget)["input_ids"]
    truncated = tokenizer.decode(ids, skip_special_tokens=True).strip()
    return truncated if truncated else sentence[:1]  # never send an empty string as the prompt


# ---------------------------------------------------------------------------
# Mask value-convention safety (see module docstring)
# ---------------------------------------------------------------------------

def binarize_mask_to_tempfile(mask_path: str, tmp_dir: Path, name: str) -> str:
    img = nib.load(mask_path)
    data = (np.asarray(img.get_fdata()) > 0.5).astype(np.int16)
    out = nib.Nifti1Image(data, img.affine, img.header)
    out_path = tmp_dir / f"{name}.nii.gz"
    nib.save(out, str(out_path))
    return str(out_path)


# ---------------------------------------------------------------------------
# Model construction (mirrors inference_demo.py's own main(), unmodified)
# ---------------------------------------------------------------------------

def build_segvol_model(args, device: torch.device):
    sys.path.insert(0, str(args.segvol_repo.resolve()))
    from segment_anything_volumetric import sam_model_registry
    from network.model import SegVol

    sam_model = sam_model_registry["vit"](args=args)
    segvol_model = SegVol(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        clip_ckpt=str(args.clip_ckpt),
        roi_size=args.spatial_size,
        patch_size=args.patch_size,
        test_mode=args.test_mode,
    ).to(device)
    segvol_model = torch.nn.DataParallel(segvol_model, device_ids=[device.index or 0])

    checkpoint = torch.load(str(args.checkpoint), map_location=device)
    state_dict = checkpoint["model"]
    # The published checkpoint's CLIP text tower was saved under an older transformers
    # version where CLIPTextModel nested its embeddings/encoder under a .text_model
    # submodule (module.text_encoder.clip_text_model.text_model.*). The transformers
    # >=4.51.0 this image actually installs (see segvol/Dockerfile's own note on why
    # not the README's pinned 4.18.0) flattened that submodule away, so the freshly-
    # built model expects module.text_encoder.clip_text_model.* directly -- strip the
    # stale ".text_model." segment so these CLIP weights still load under strict=True
    # instead of hitting a hard state_dict mismatch. Doesn't touch any non-CLIP key.
    remapped = 0
    for key in list(state_dict.keys()):
        if ".clip_text_model.text_model." in key:
            state_dict[key.replace(".clip_text_model.text_model.", ".clip_text_model.")] = state_dict.pop(key)
            remapped += 1
    if remapped:
        logger.info(f"Remapped {remapped} CLIP text-tower key(s) for the installed transformers version")
    # position_ids is a fixed arange buffer, not a learned weight -- older transformers
    # persisted it in state_dict, but transformers>=4.51.0 registers it non-persistent
    # (CLIPTextEmbeddings recomputes it at forward time), so a freshly-built model has
    # no such key at all and strict=True would otherwise reject it as unexpected.
    dropped = [k for k in state_dict if k.endswith(".clip_text_model.embeddings.position_ids")]
    for key in dropped:
        del state_dict[key]
    if dropped:
        logger.info(f"Dropped {len(dropped)} non-persistent position_ids buffer key(s) from the checkpoint")
    segvol_model.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded SegVol checkpoint {args.checkpoint} (epoch {checkpoint.get('epoch')})")
    return segvol_model


# ---------------------------------------------------------------------------
# Scoring / viz helpers
# ---------------------------------------------------------------------------

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
    image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, sentence: str, dice: float,
    out_path: Path, vmin: float = -150.0, vmax: float = 250.0,
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
        pred_mask[d, :, :].any() or pred_mask[:, h, :].any() or pred_mask[:, :, w].any()
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
                        framealpha=0.4, facecolor="black", labelcolor="white", edgecolor="gray")

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


def _write_summary(output_path: Path, checkpoint: str, hit_threshold: float,
                    records: list[dict], skipped: list[dict]) -> dict:
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
    sys.path.insert(0, str(args.segvol_repo.resolve()))
    from data_process.demo_data_process import process_ct_gt
    from inference_demo import zoom_in_zoom_out

    with open(args.manifest) as f:
        samples = json.load(f)
    kept = [s for s in samples if s.get("sentence")]
    n_dropped = len(samples) - len(kept)
    if n_dropped:
        logger.warning(f"Dropping {n_dropped}/{len(samples)} sample(s) with no sentence: "
                        f"{[s['mask'] for s in samples if not s.get('sentence')]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    logger.info(f"Loading SegVol from {args.segvol_repo} (checkpoint={args.checkpoint})")
    segvol_model = build_segvol_model(args, device)

    clip_tokenizer = build_clip_tokenizer(args.clip_ckpt)
    budget = max_sentence_tokens(clip_tokenizer)
    logger.info(f"Truncating referring expressions to {budget} CLIP tokens before templating")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.visualize:
        viz_dir = Path(args.viz_output_dir) if args.viz_output_dir else output_path.parent / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing visualizations -> {viz_dir}")

    records: list[dict] = []
    skipped: list[dict] = []
    tmp_dir_ctx = tempfile.TemporaryDirectory()
    tmp_dir = Path(tmp_dir_ctx.name)

    for i, s in enumerate(kept):
        image_path = f"{args.image_dir}/{s['image']}"
        mask_path = f"{args.mask_dir}/{s['mask']}"
        sentence = s["sentence"].strip()
        truncated = truncate_sentence_for_clip(clip_tokenizer, sentence, budget)

        try:
            safe_mask_path = binarize_mask_to_tempfile(mask_path, tmp_dir, _safe_filename(s["mask"]))
            item = process_ct_gt(image_path, safe_mask_path, [truncated], args.spatial_size)
            image = item["image"].float()
            gt3D = item["label"]
            image_resize = item["zoom_out_image"].float()
            gt3D_resize = item["zoom_out_label"]

            logits_labels_record = zoom_in_zoom_out(
                args, segvol_model,
                image.unsqueeze(0), image_resize.unsqueeze(0),
                gt3D.unsqueeze(0), gt3D_resize.unsqueeze(0),
                categories=[truncated],
            )
            if truncated not in logits_labels_record:
                raise ValueError("GT mask was empty after preprocessing (zoom_in_zoom_out skipped it)")

            dice_tensor, image_single, _points, _box, logits_global_single, label_single = logits_labels_record[truncated]
            dice = float(dice_tensor.item())
            pred_mask = (torch.sigmoid(logits_global_single) > 0.5).cpu().numpy().astype(np.float32)
            gt_mask = label_single.cpu().numpy().astype(np.float32)
            image_np = image_single.cpu().numpy()

            cat = s.get("finding") or "Unknown"
            records.append({"id": s["mask"], "category": cat, "dice": dice,
                             "hit": bool(dice >= args.hit_threshold)})

            if args.visualize:
                cat_dir = viz_dir / cat
                cat_dir.mkdir(parents=True, exist_ok=True)
                _save_figure(image_np, gt_mask, pred_mask, sentence, dice,
                             cat_dir / f"{_safe_filename(s['mask'])}.png")
        except Exception as e:
            logger.warning(f"SKIP mask {s['mask']}: {e}")
            skipped.append({"image": s["image"], "mask": s["mask"], "reason": str(e)})

        if i % 25 == 0:
            _write_summary(output_path, str(args.checkpoint), args.hit_threshold, records, skipped)
        torch.cuda.empty_cache()

    tmp_dir_ctx.cleanup()
    summary = _write_summary(output_path, str(args.checkpoint), args.hit_threshold, records, skipped)

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


if __name__ == "__main__":
    main()
