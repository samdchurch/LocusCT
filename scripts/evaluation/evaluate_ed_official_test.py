#!/usr/bin/env python3
"""
Score a Grounder checkpoint on the official ED held-out test set
(official_splits/ed_official_test_data.json), reporting Dice and hit rate
overall and broken down per finding category. Also writes one GT/pred
overlay PNG per case (axial/coronal/sagittal slices, green GT contour, red
predicted contour) to <output's parent>/viz/<category>/, matching
visualize.py's style. Pass --no-visualize to skip this.

The manifest's image/mask paths are relative to two different roots than the
ones configs/default.yaml points at by default: images resolve against
inhouse_abdominal_ct/nifti_resampled, and masks resolve against
inhouse_abdominal_ct/ED_TEST_SET_resampled -- both on the same 1.5x1.5x3.0mm/
352x352x180 grid the model trains on, once resample_and_crop.py and
resample_masks.py have run on these accessions. Pass --image-dir/--mask-dir
to point at wherever those actually resolve on the machine this runs on. One
example (KIDNEYSTONE/CASE0000000) has sentence=null in the manifest (no
matching GSPS annotation was found) and is skipped.

Pass --save_masks (plus --raw-image-dir, e.g. .../inhouse_abdominal_ct/nifti) to
write each predicted mask to <output's parent>/predicted_masks/, resampled
back onto that case's original, pre-resample_and_crop.py raw image grid
(not just dumped in the model's 352x352x180 working resolution) -- see
_canonical_mask_affine()/_save_masks() for how the affine is carried through
correctly.

Usage
-----
    python evaluate_ed_official_test.py --config configs/default.yaml --checkpoint runs/default/checkpoints/best.pt \
        --image-dir /path/to/data/inhouse_abdominal_ct/nifti_resampled \
        --mask-dir /path/to/data/inhouse_abdominal_ct/ED_TEST_SET_resampled \
        --save_masks --raw-image-dir /path/to/data/inhouse_abdominal_ct/nifti
"""

import argparse
import copy
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
import yaml
from matplotlib.lines import Line2D
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import build_dataloader
from models.grounder import Grounder
from train import apply_overrides
from training.trainer import _trim_padding
from utils.metrics import dice_score, iou_score, precision_recall

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "ed_official_test_data.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-dir", required=True,
                         help="Base dir the manifest's relative image paths resolve against")
    parser.add_argument("--mask-dir", required=True,
                         help="Base dir the manifest's relative mask paths resolve against")
    parser.add_argument("--raw-image-dir", default=None,
                         help="Base dir of the original, pre-resample_and_crop.py NIfTI files (e.g. "
                              ".../inhouse_abdominal_ct/nifti). Only needed with --save_masks -- predicted masks are "
                              "resampled back onto each case's original raw grid, not saved in model space.")
    parser.add_argument("--output", default="outputs/eval/ed_official_test/results.json")
    parser.add_argument("--hit-threshold", type=float, default=0.1,
                         help="Dice value at/above which a sample counts as a 'hit' (matches training's convention)")
    parser.add_argument("--save_masks", action="store_true",
                         help="Save predicted masks as .nii.gz, resampled onto each case's original raw image "
                              "grid (requires nibabel and --raw-image-dir)")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false",
                         help="Skip writing per-case GT/pred overlay PNGs (on by default)")
    parser.add_argument("--viz-output-dir", default=None,
                         help="Where to write visualization PNGs. Default: <output's parent>/viz")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Dot-notation config overrides, e.g. model.text_encoder_name=/path/to/model",
    )
    return parser.parse_args()


def _build_finding_lookup(manifest_path: Path) -> dict[str, str]:
    """Groups by the manifest's own "finding" field rather than parsing the mask
    path's leading folder -- LocusBench's mask paths are prefixed with "masks/"
    (unlike the old official manifest's bare "CATEGORY/accession/..."), so path
    parsing no longer recovers the category. Matches
    evaluate_onc_official_test.py's _build_finding_lookup() exactly."""
    with open(manifest_path) as f:
        samples = json.load(f)
    return {s["mask"]: (s.get("finding") or "Unknown") for s in samples}


def _filter_manifest(manifest_path: Path, output_dir: Path) -> Path:
    """Drop entries with sentence=null (can't tokenize) and write a filtered copy.

    Returns the path to the filtered manifest for build_dataloader to consume.
    """
    with open(manifest_path) as f:
        samples = json.load(f)

    kept = [s for s in samples if s.get("sentence")]
    n_dropped = len(samples) - len(kept)
    if n_dropped:
        logger.warning(f"Dropping {n_dropped}/{len(samples)} sample(s) with no sentence: "
                        f"{[s['mask'] for s in samples if not s.get('sentence')]}")

    filtered_path = output_dir / "ed_official_test_data.filtered.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(filtered_path, "w") as f:
        json.dump(kept, f)
    return filtered_path


def _build_sentence_lookup(manifest_path: Path) -> dict[str, str]:
    with open(manifest_path) as f:
        samples = json.load(f)
    return {s["mask"]: s["sentence"] for s in samples}


def _build_image_lookup(manifest_path: Path) -> dict[str, str]:
    with open(manifest_path) as f:
        samples = json.load(f)
    return {s["mask"]: s["image"] for s in samples}


def _find_series_file(accession_dir: Path, series: int) -> Path | None:
    if not accession_dir.is_dir():
        return None
    for f in sorted(accession_dir.glob("*.nii.gz")):
        prefix = f.name.split("_")[0]
        if prefix.isdigit() and int(prefix) == series:
            return f
    return None


def _mask_centroid(mask: np.ndarray) -> tuple[int, int, int]:
    """(D, H, W) binary mask → (d, h, w) centroid; falls back to volume center."""
    coords = np.argwhere(mask > 0.5)
    if len(coords) == 0:
        return tuple(s // 2 for s in mask.shape)
    d, h, w = coords.mean(axis=0).astype(int)
    d = int(np.clip(d, 0, mask.shape[0] - 1))
    h = int(np.clip(h, 0, mask.shape[1] - 1))
    w = int(np.clip(w, 0, mask.shape[2] - 1))
    return d, h, w


def _trim_volume(vol: np.ndarray, pad_amounts: torch.Tensor) -> np.ndarray:
    """Remove end-padding from a (D, H, W) numpy array."""
    pad_D, pad_H, pad_W = (int(p) for p in pad_amounts)
    D = vol.shape[0] - pad_D if pad_D > 0 else vol.shape[0]
    H = vol.shape[1] - pad_H if pad_H > 0 else vol.shape[1]
    W = vol.shape[2] - pad_W if pad_W > 0 else vol.shape[2]
    return vol[:D, :H, :W]


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
    vmin: float,
    vmax: float,
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
            if row_idx == 0 and label == "Axial":
                pct = np.percentile(img_sl, [1, 5, 50, 95, 99])
                logger.info(f"  imshow input: dtype={img_sl.dtype}  shape={img_sl.shape}  "
                            f"min={img_sl.min():.4f}  max={img_sl.max():.4f}  "
                            f"p1={pct[0]:.4f} p5={pct[1]:.4f} p50={pct[2]:.4f} p95={pct[3]:.4f} p99={pct[4]:.4f}  "
                            f"vmin={vmin:.4f}  vmax={vmax:.4f}")
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
    iou_vals = [r["iou"] for r in records]
    prec_vals = [r["precision"] for r in records]
    rec_vals = [r["recall"] for r in records]
    return {
        "n_samples": len(records),
        "dice_mean": float(np.mean(dice_vals)),
        "dice_std": float(np.std(dice_vals)),
        "iou_mean": float(np.mean(iou_vals)),
        "iou_std": float(np.std(iou_vals)),
        "precision_mean": float(np.mean(prec_vals)),
        "recall_mean": float(np.mean(rec_vals)),
        "hit_rate": float(np.mean([r["hit"] for r in records])),
    }


def main() -> None:
    args = parse_args()
    if args.save_masks and not args.raw_image_dir:
        sys.exit("--save_masks requires --raw-image-dir (base dir of the original, pre-resample_and_crop.py "
                  "NIfTI files) to resample predicted masks back onto each case's original grid.")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = cfg["inference"].get("threshold", 0.5)

    # Mirror train.py's flag: whichever embedding_cache setting the checkpoint
    # was trained under, matching it here just avoids paying to load the 8B
    # backbone when we don't need to -- per-stage projections now live in the
    # UNet's cross-attention modules and train regardless of this flag, so
    # live vs. cached text features are equivalent, not just "safe."
    use_cached_text = bool(cfg["data"].get("embedding_cache"))

    model = Grounder(
        text_encoder_name=cfg["model"]["text_encoder_name"],
        text_proj_dim=cfg["model"]["text_proj_dim"],
        freeze_text_encoder=True,
        finetune_last_n_layers=0,
        unet_base_channels=cfg["model"]["unet_base_channels"],
        unet_channel_mult=cfg["model"]["unet_channel_mult"],
        num_heads=cfg["model"]["num_heads"],
        target_q_tokens=cfg["model"]["target_q_tokens"],
        dropout=0.0,
        in_channels=3 if cfg["data"].get("multi_window") else 1,
        spatial_size=tuple(cfg["data"]["spatial_size"]),
        fusion_type=cfg["model"].get("fusion_type", "cross_attention"),
        voxtell_guidance_dim=cfg["model"].get("voxtell", {}).get("guidance_dim", 32),
        voxtell_prompt_decoder_dim=cfg["model"].get("voxtell", {}).get("prompt_decoder_dim", 256),
        voxtell_prompt_decoder_layers=cfg["model"].get("voxtell", {}).get("prompt_decoder_layers", 6),
        voxtell_prompt_decoder_heads=cfg["model"].get("voxtell", {}).get("prompt_decoder_heads", 8),
        load_text_backbone=not use_cached_text,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    output_path = Path(args.output)
    filtered_manifest = _filter_manifest(args.manifest, output_path.parent or Path("."))

    cfg_ed = copy.deepcopy(cfg)
    cfg_ed["data"]["image_dir"] = args.image_dir
    cfg_ed["data"]["mask_dir"] = args.mask_dir
    loader = build_dataloader(str(filtered_manifest), cfg_ed, split="test", num_workers=0)

    finding_lookup = _build_finding_lookup(filtered_manifest)

    if args.save_masks:
        image_lookup = _build_image_lookup(filtered_manifest)
        image_dir = Path(args.image_dir)
        raw_image_dir = Path(args.raw_image_dir)

    if args.visualize:
        viz_dir = Path(args.viz_output_dir) if args.viz_output_dir else (output_path.parent or Path(".")) / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        sentence_lookup = _build_sentence_lookup(filtered_manifest)
        # Convert HU display window to normalized [-1, 1] space, same as visualize.py
        hu_min, hu_max = cfg["data"]["hu_min"], cfg["data"]["hu_max"]
        def _hu_to_norm(hu: float) -> float:
            return (hu - hu_min) / (hu_max - hu_min) * 2.0 - 1.0
        viz_vmin, viz_vmax = _hu_to_norm(-150), _hu_to_norm(250)
        logger.info(f"Writing visualizations → {viz_dir}")

    records: list[dict] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating ED official test set"):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if use_cached_text:
                    text_feats = batch["text_feats"].to(device)
                    text_padding_mask = batch["text_padding_mask"].to(device)
                    logits = model(image, text_feats=text_feats, text_padding_mask=text_padding_mask)
                else:
                    input_ids = batch["input_ids"].to(device)
                    attn_mask = batch["attention_mask"].to(device)
                    logits = model(image, input_ids, attn_mask)

            logits, mask = _trim_padding(logits, mask, batch["pad_amounts"])

            dices = dice_score(logits, mask, threshold=threshold, from_logits=True)
            ious = iou_score(logits, mask, threshold=threshold, from_logits=True)
            prec, rec = precision_recall(logits, mask, threshold=threshold, from_logits=True)

            for i, sample_id in enumerate(batch["id"]):
                dice = dices[i].item()
                records.append(
                    {
                        "id": sample_id,
                        "category": finding_lookup.get(sample_id, "Unknown"),
                        "dice": dice,
                        "iou": ious[i].item(),
                        "precision": prec[i].item(),
                        "recall": rec[i].item(),
                        "hit": bool(dice >= args.hit_threshold),
                    }
                )

                if args.visualize:
                    image_np = _trim_volume(image[i, 0].cpu().float().numpy(), batch["pad_amounts"][i])
                    gt_np = mask[i, 0].cpu().float().numpy()
                    pred_np = (torch.sigmoid(logits[i, 0]).cpu().float().numpy() > threshold).astype(np.float32)
                    sentence = sentence_lookup.get(sample_id, sample_id)
                    cat_dir = viz_dir / finding_lookup.get(sample_id, "Unknown")
                    cat_dir.mkdir(parents=True, exist_ok=True)
                    _save_figure(
                        image_np, gt_np, pred_np, sentence, dice,
                        cat_dir / f"{_safe_filename(sample_id)}.png",
                        vmin=viz_vmin, vmax=viz_vmax,
                    )

            if args.save_masks:
                _save_masks(logits, batch, threshold, args.output, image_lookup, image_dir, raw_image_dir)

    by_category = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r)

    summary = {
        "checkpoint": args.checkpoint,
        "hit_threshold": args.hit_threshold,
        "overall": summarize(records),
        "by_category": {cat: summarize(recs) for cat, recs in sorted(by_category.items())},
        "per_sample": records,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    overall = summary["overall"]
    logger.info(
        f"Results → {output_path}\n"
        f"  Overall ({overall['n_samples']} samples):\n"
        f"    Dice: {overall['dice_mean']:.4f} ± {overall['dice_std']:.4f}   "
        f"IoU: {overall['iou_mean']:.4f} ± {overall['iou_std']:.4f}   "
        f"Prec: {overall['precision_mean']:.4f}  Rec: {overall['recall_mean']:.4f}   "
        f"Hit: {overall['hit_rate']:.4f}"
    )
    logger.info("  By category:")
    for cat, s in summary["by_category"].items():
        logger.info(
            f"    {cat:<15} n={s['n_samples']:<4} "
            f"Dice={s['dice_mean']:.4f}±{s['dice_std']:.4f}  Hit={s['hit_rate']:.4f}"
        )
    if args.visualize:
        logger.info(f"Wrote {len(records)} visualization(s) → {viz_dir}")


def _canonical_mask_affine(resampled_img) -> np.ndarray:
    """
    Affine for an array in the exact (D, H, W) layout load_nifti_canonical()
    produces from `resampled_img`, so a predicted mask (which is in that
    layout) can be paired with a correct, real-world-aligned affine without
    hand-inverting as_closest_canonical's reorientation -- nibabel already
    computes a correct affine for the RAS-canonical (X, Y, Z) array;
    load_nifti_canonical's remaining steps (transpose(2, 1, 0), then
    _normalize_depth_axis's conditional moveaxis) are pure axis
    permutations, no flips, so their effect on the affine is just the
    matching column permutation.
    """
    import nibabel as nib

    canonical = nib.as_closest_canonical(resampled_img)
    affine = canonical.affine.copy()
    affine[:3, :3] = affine[:3, 2::-1]  # transpose(2, 1, 0) -- reverse only the 3 spatial columns, not the translation column

    s0, s1, s2 = canonical.shape[::-1]  # shape after the transpose above
    if s1 == s2:
        perm = (0, 1, 2)
    elif s0 == s2:
        perm = (1, 0, 2)  # mirrors _normalize_depth_axis's moveaxis(arr, 1, 0)
    elif s0 == s1:
        perm = (2, 0, 1)  # mirrors _normalize_depth_axis's moveaxis(arr, 2, 0)
    else:
        perm = (0, 1, 2)  # mirrors _normalize_depth_axis's "can't determine" fallback
    affine[:3, :3] = affine[:3, perm]
    return affine


def _save_masks(
    logits: torch.Tensor,
    batch: dict,
    threshold: float,
    output_path: str,
    image_lookup: dict[str, str],
    image_dir: Path,
    raw_image_dir: Path,
) -> None:
    """
    Saves each predicted mask resampled back onto its case's original
    (pre-resample_and_crop.py) raw NIfTI grid, correctly registered via
    nibabel affine math -- not just a same-shape array dump with an
    identity affine. The model's output lives on the "nifti_resampled"
    grid, whose on-disk affine already correctly maps to world space (see
    resample_and_crop.py's build_affine()); _canonical_mask_affine() carries
    that affine through load_nifti_canonical's reordering so the predicted
    mask is genuinely spatially registered, then
    nibabel.processing.resample_from_to (order=0, nearest-neighbor -- this
    is a binary mask) resamples it onto the raw image's own grid/affine.
    Falls back to an identity-affine, model-space dump (the old behavior)
    if the raw image or its series number can't be resolved.
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to

    out_dir = Path(output_path).parent / "predicted_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = torch.sigmoid(logits).cpu().float().numpy()

    for i, sample_id in enumerate(batch["id"]):
        mask_np = (probs[i, 0] > threshold).astype(np.uint8)
        safe_id = sample_id.replace("/", "_")

        def _save_fallback(reason: str) -> None:
            logger.warning(f"{sample_id}: {reason}; saving in model space with identity affine")
            nib.save(nib.Nifti1Image(mask_np, np.eye(4)), out_dir / f"{safe_id}_pred.nii.gz")

        image_rel = image_lookup.get(sample_id)
        if image_rel is None:
            _save_fallback("no image path known for this sample")
            continue

        series_str = Path(image_rel).name.split("_")[0]
        if not series_str.isdigit():
            _save_fallback(f"can't parse series number from image path {image_rel!r}")
            continue

        accession = Path(image_rel).parts[-2]
        raw_path = _find_series_file(raw_image_dir / accession, int(series_str))
        if raw_path is None:
            _save_fallback(f"no raw series {series_str} file found under {raw_image_dir / accession}")
            continue

        resampled_img = nib.load(str(image_dir / image_rel))
        mask_img = nib.Nifti1Image(mask_np, _canonical_mask_affine(resampled_img))
        raw_img = nib.load(str(raw_path))
        mask_on_raw_grid = resample_from_to(mask_img, raw_img, order=0, mode="constant", cval=0)
        nib.save(mask_on_raw_grid, out_dir / f"{safe_id}_pred.nii.gz")


if __name__ == "__main__":
    main()
