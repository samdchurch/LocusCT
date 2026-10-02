#!/usr/bin/env python3
"""
Scores OUR fine-tuned VoxTell checkpoint (finetune_voxtell.py's
model_state_dict format) on the official ED held-out test set
(official_splits/ed_official_test_data.json), with real full-volume
sliding-window inference -- 192^3 tiles, since VoxTellModel's
positional-encoding buffer is fixed to that patch size (see
finetune_voxtell.py's module docstring) and can't take a full volume in one
shot. This replaces finetune_voxtell.py's fast single-patch validation proxy
with a real held-out number, per that script's own docstring recommendation.

NOT a drop-in replacement for evaluate_voxtell_ed.py's VoxTellPredictor path:
that expects the *original* checkpoint format (plans.json + fold_0/
checkpoint_final.pth with a "network_weights" key) and its own preprocessing
built for VoxTell's original single raw-HU channel. Our fine-tuned checkpoint
is a different format (model_state_dict) and -- by default -- a different
input (3-channel lung/soft-tissue/bone windowing). This script instead
reuses finetune_voxtell.py's own model construction (build_voxtell_model),
preprocessing (apply_windows, load_nifti_canonical), and text encoding
(load_text_backbone/embed_sentences), so the model is evaluated exactly the
way it was trained. Reporting/visualization (category_of, _save_figure,
_safe_filename, _write_summary) are imported directly from
evaluate_voxtell_ed.py -- nothing duplicated.

Tiles are non-overlapping by default (--tile-overlap 0.0) -- simpler, and a
big step up from the single-patch training proxy, but tile-boundary
artifacts are possible. Pass --tile-overlap (e.g. 0.5) for nnU-Net-style
overlapping/Gaussian-weighted windows instead, at the cost of roughly
1/(1-overlap)^3 more forward passes per volume -- see
sliding_window_predict's docstring.

Pass --save-masks to write each predicted mask as .nii.gz to <output's
parent>/predicted_masks/, thresholded at 0.5. Without --raw-image-dir, masks
are saved directly on --image-dir's own grid (the grid the model actually
predicted on -- correctly registered via that image's own affine, see
_save_predicted_mask). With --raw-image-dir (e.g. .../inhouse_abdominal_ct/nifti),
masks are additionally resampled onto each case's original, pre-resample_and_
crop.py raw grid instead, reusing evaluate_ed_official_test.py's
_canonical_mask_affine/_find_series_file -- same convention as that script's
own --save_masks.

Usage
-----
    python evaluate_finetuned_voxtell_ed.py --checkpoint runs/voxtell_finetune/checkpoints/best.pt
    python evaluate_finetuned_voxtell_ed.py --checkpoint ... --visualize
    python evaluate_finetuned_voxtell_ed.py --checkpoint ... --save-masks \
        --raw-image-dir /path/to/data/inhouse_abdominal_ct/nifti
"""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import load_nifti_canonical
from evaluate_ed_official_test import _canonical_mask_affine, _find_series_file
from evaluate_voxtell_ed import _safe_filename, _save_figure, _write_summary, category_of
from finetune_voxtell import (
    DEFAULT_MODEL_DIR,
    DEFAULT_TEXT_ENCODER,
    PATCH_SIZE,
    apply_windows,
    build_voxtell_model,
    embed_sentences,
    load_text_backbone,
)
from utils.metrics import dice_score

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "ed_official_test_data.json"
# Native (pre-resample) roots, same convention as finetune_voxtell.py -- the
# ED official test set's masks live under a separate ED_TEST_SET/ tree, not
# the main dataset's labels/ (see resample_masks.py's ED_TEST_ROOT).
DEFAULT_IMAGE_DIR = "/path/to/data/inhouse_abdominal_ct/nifti"
DEFAULT_MASK_DIR = "/path/to/data/inhouse_abdominal_ct/ED_TEST_SET"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                         help="finetune_voxtell.py checkpoint (e.g. runs/voxtell_finetune/checkpoints/best.pt)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR,
                         help="Base dir the manifest's relative 'image' paths resolve against")
    parser.add_argument("--mask-dir", default=DEFAULT_MASK_DIR,
                         help="Base dir the manifest's relative 'mask' paths resolve against")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                         help="VoxTell architecture/plans dir (plans.json) -- weights come from --checkpoint, not this")
    parser.add_argument("--text-encoder", default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--multi-window", dest="multi_window", action="store_true", default=False,
                         help="Must match how --checkpoint was fine-tuned (default: single Z-score-normalized channel)")
    parser.add_argument("--patch-size", type=int, default=192,
                         help="Must match how --checkpoint was trained (finetune_voxtell.py's --patch-size).")
    parser.add_argument("--num-maskformer-stages", type=int, default=5,
                         help="Must match how --checkpoint was trained (finetune_voxtell.py's build_voxtell_model).")
    parser.add_argument("--decoder-layer", type=int, default=4,
                         help="Must match how --checkpoint was trained (finetune_voxtell.py's build_voxtell_model).")
    parser.add_argument("--text-embedding-dim", type=int, default=2560,
                         help="Must match how --checkpoint was trained -- 2560 for Qwen3-Embedding-4B (default), "
                              "4096 for Qwen3-Embedding-8B.")
    parser.add_argument("--tile-overlap", type=float, default=0.0,
                         help="Fraction (0.0-<1.0) of patch_size neighboring sliding-window tiles overlap by, "
                              "blended via a Gaussian weight map (nnU-Net-style). 0.0 (default): tiles are still "
                              "tail-aligned to the volume boundary (see _tile_starts) but otherwise not "
                              "overlapping. Higher values trade ~1/(1-overlap)^3 more forward passes for softer "
                              "tile-boundary artifacts.")
    parser.add_argument("--output", default="outputs/eval/voxtell_finetuned_ed/results.json")
    parser.add_argument("--hit-threshold", type=float, default=0.1,
                         help="Dice value at/above which a sample counts as a 'hit'")
    parser.add_argument("--visualize", dest="visualize", action="store_true", default=False,
                         help="Write per-case GT/pred overlay PNGs (off by default)")
    parser.add_argument("--viz-output-dir", default=None,
                         help="Where to write visualization PNGs. Default: <output's parent>/viz")
    parser.add_argument("--save-masks", dest="save_masks", action="store_true", default=False,
                         help="Write each predicted mask (thresholded at 0.5) as .nii.gz to "
                              "<output's parent>/predicted_masks/ (off by default)")
    parser.add_argument("--raw-image-dir", default=None,
                         help="Base dir of the original, pre-resample_and_crop.py raw NIfTI files "
                              "(e.g. .../inhouse_abdominal_ct/nifti). If given with --save-masks, predicted masks "
                              "are additionally resampled onto that case's original raw grid instead of being "
                              "saved directly on --image-dir's working grid.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _tile_starts(volume_size: int, patch_size: int, step_fraction: float) -> list[int]:
    """Start offsets covering [0, volume_size) with patch_size-wide tiles, spaced as close
    to patch_size * step_fraction apart as an even division of the volume allows, with the
    last tile pulled in to end exactly at volume_size -- same 'steps' construction nnU-Net's
    own sliding-window inference uses. Actual spacing may end up tighter (more overlap) than
    requested to make the tiling come out even; it is never looser. step_fraction=1.0 (tile_
    overlap=0.0) still uses this -- unlike a naive fixed-stride tiling, it never leaves the
    final tile mostly hanging off the volume edge into padding."""
    if volume_size <= patch_size:
        return [0]
    target_step = patch_size * step_fraction
    num_steps = int(math.ceil((volume_size - patch_size) / target_step)) + 1
    actual_step = (volume_size - patch_size) / (num_steps - 1)
    return [int(round(actual_step * i)) for i in range(num_steps)]


def _gaussian_weight_map(patch_size: tuple[int, int, int], sigma_scale: float = 1.0 / 8) -> torch.Tensor:
    """Isotropic-per-axis Gaussian centered on the patch (same construction nnU-Net's own
    sliding-window inference uses), so blending overlapping tiles weights tile-center
    predictions -- which have more surrounding context than tile-edge ones -- more heavily.
    Clamped away from 0 so a voxel touched by only one tile isn't near-zeroed by it."""
    coords = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in patch_size], indexing="ij")
    weight = torch.ones(patch_size, dtype=torch.float32)
    for size, coord in zip(patch_size, coords):
        center = (size - 1) / 2
        sigma = size * sigma_scale
        weight *= torch.exp(-((coord - center) ** 2) / (2 * sigma ** 2))
    return (weight / weight.max()).clamp(min=1e-4)


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    text_embeddings: torch.Tensor,
    device: torch.device,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
    tile_overlap: float = 0.0,
) -> np.ndarray:
    """
    image: (C, D, H, W) preprocessed (apply_windows'd) volume.
    text_embeddings: (N, 1, text_dim) -- N referring expressions for this image.
    Returns (N, D, H, W) float32 probabilities, via patch_size tiles covering the whole
    volume (see _tile_starts) -- each tile location is a single forward pass batched over
    all N sentences, so the image is only re-encoded once per tile, not once per (tile,
    sentence). Only padded (with -1.0 "air", matching foreground_patch's convention) if the
    volume is smaller than patch_size in some dim.

    tile_overlap=0.0 (default): tiles are tail-aligned (see _tile_starts) but not otherwise
    overlapping -- one forward pass covers each voxel almost everywhere.
    tile_overlap>0 (up to just under 1.0): tiles additionally overlap by (approximately)
    that fraction of patch_size, blended via _gaussian_weight_map instead of a flat average,
    trading ~1/(1-tile_overlap)^3 more forward passes for softer tile-boundary artifacts.
    """
    orig_D, orig_H, orig_W = image.shape[1:]
    pd, ph, pw = patch_size
    if orig_D < pd or orig_H < ph or orig_W < pw:
        image = F.pad(image, (0, max(0, pw - orig_W), 0, max(0, ph - orig_H), 0, max(0, pd - orig_D)),
                       mode="constant", value=-1.0)
    _, D, H, W = image.shape

    step_fraction = 1.0 - tile_overlap
    d_starts = _tile_starts(D, pd, step_fraction)
    h_starts = _tile_starts(H, ph, step_fraction)
    w_starts = _tile_starts(W, pw, step_fraction)

    n = text_embeddings.shape[0]
    weight_map = (
        _gaussian_weight_map(patch_size) if tile_overlap > 0 else torch.ones(patch_size, dtype=torch.float32)
    )
    probs = torch.zeros((n, D, H, W), dtype=torch.float32)
    weight_sum = torch.zeros((D, H, W), dtype=torch.float32)

    for d0 in d_starts:
        for h0 in h_starts:
            for w0 in w_starts:
                patch = image[:, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw].unsqueeze(0).to(device)
                patch_batch = patch.expand(n, -1, -1, -1, -1)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(patch_batch, text_embeddings.to(device))
                tile_probs = torch.sigmoid(logits[:, 0]).float().cpu()
                probs[:, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw] += tile_probs * weight_map
                weight_sum[d0:d0 + pd, h0:h0 + ph, w0:w0 + pw] += weight_map

    probs /= weight_sum.clamp(min=1e-8)
    return probs[:, :orig_D, :orig_H, :orig_W].numpy()


def _save_predicted_mask(
    pr: np.ndarray,
    threshold: float,
    sample_id: str,
    image_rel: str,
    image_dir: Path,
    raw_image_dir: Path | None,
    out_dir: Path,
) -> None:
    """
    Saves one predicted mask as a correctly-registered NIfTI, reusing
    evaluate_ed_official_test.py's _canonical_mask_affine/_find_series_file
    (nothing duplicated). Without raw_image_dir, saves directly on the
    --image-dir grid the model actually predicted on -- already correctly
    registered via that image's own affine, no resampling needed. With
    raw_image_dir, additionally resamples (nearest-neighbor, since this is a
    binary mask) onto that case's original pre-resample raw grid, same as
    evaluate_ed_official_test.py's --save_masks.
    """
    import nibabel as nib

    mask_np = (pr > threshold).astype(np.uint8)
    safe_id = sample_id.replace("/", "_")
    resampled_img = nib.load(str(image_dir / image_rel))
    mask_img = nib.Nifti1Image(mask_np, _canonical_mask_affine(resampled_img))

    if raw_image_dir is not None:
        series_str = Path(image_rel).name.split("_")[0]
        accession = Path(image_rel).parts[-2]
        raw_path = _find_series_file(raw_image_dir / accession, int(series_str)) if series_str.isdigit() else None
        if raw_path is not None:
            from nibabel.processing import resample_from_to
            mask_img = resample_from_to(mask_img, nib.load(str(raw_path)), order=0, mode="constant", cval=0)
        else:
            logger.warning(f"{sample_id}: couldn't resolve a raw series file for image {image_rel!r}; "
                            f"saving on the --image-dir working grid instead")

    nib.save(mask_img, out_dir / f"{safe_id}_pred.nii.gz")


def main() -> None:
    args = parse_args()
    args.patch_size = (args.patch_size, args.patch_size, args.patch_size)
    device = torch.device(args.device)

    with open(args.manifest) as f:
        samples = json.load(f)
    kept = [s for s in samples if s.get("sentence")]
    n_dropped = len(samples) - len(kept)
    if n_dropped:
        logger.warning(f"Dropping {n_dropped}/{len(samples)} sample(s) with no sentence: "
                        f"{[s['mask'] for s in samples if not s.get('sentence')]}")

    num_channels = 3 if args.multi_window else 1
    logger.info(f"Building VoxTell model (input_channels={num_channels}) from {args.model_dir}")
    model = build_voxtell_model(
        Path(args.model_dir), num_channels,
        num_maskformer_stages=args.num_maskformer_stages, decoder_layer=args.decoder_layer,
        text_embedding_dim=args.text_embedding_dim,
    ).to(device)

    logger.info(f"Loading fine-tuned weights from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info(f"Loading frozen text backbone {args.text_encoder}")
    tokenizer, text_backbone = load_text_backbone(args.text_encoder, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.visualize:
        viz_dir = Path(args.viz_output_dir) if args.viz_output_dir else output_path.parent / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing visualizations -> {viz_dir}")

    if args.save_masks:
        masks_out_dir = output_path.parent / "predicted_masks"
        masks_out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing predicted masks -> {masks_out_dir}")

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    raw_image_dir = Path(args.raw_image_dir) if args.raw_image_dir else None

    by_image: dict[str, list[dict]] = defaultdict(list)
    for s in kept:
        by_image[s["image"]].append(s)

    records: list[dict] = []
    skipped: list[dict] = []

    try:
        for group_idx, (image_rel, group) in enumerate(
            tqdm(sorted(by_image.items()), desc="Evaluating ED official test set (fine-tuned VoxTell)")
        ):
            try:
                image_np = load_nifti_canonical(str(image_dir / image_rel))  # (D, H, W) raw HU
                windowed = apply_windows(image_np, args.multi_window)
                image_t = torch.from_numpy(windowed).float()
                sentences = [s["sentence"] for s in group]
                text_embedding = embed_sentences(tokenizer, text_backbone, sentences, device)
                probs = sliding_window_predict(model, image_t, text_embedding, device,
                                                patch_size=args.patch_size,
                                                tile_overlap=args.tile_overlap)  # (N, D, H, W)
            except Exception as e:
                logger.warning(f"SKIP image group {image_rel}: {e}")
                skipped.extend({"image": image_rel, "mask": s["mask"], "reason": str(e)} for s in group)
                continue

            for i, s in enumerate(group):
                mask_rel = s["mask"]
                try:
                    gt = (load_nifti_canonical(str(mask_dir / mask_rel)) > 0.5).astype(np.float32)
                    pr = probs[i].astype(np.float32)
                    if gt.shape != pr.shape:
                        raise ValueError(f"image/mask shape mismatch: pred={pr.shape} gt={gt.shape} "
                                          f"(mask likely drawn on a different reconstruction/series than "
                                          f"the manifest's 'image' field points to)")

                    dice = dice_score(
                        torch.from_numpy(pr[None]), torch.from_numpy(gt[None]),
                        threshold=0.5, smooth=1.0, from_logits=False,
                    )[0].item()

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

                    if args.save_masks:
                        _save_predicted_mask(pr, 0.5, mask_rel, image_rel, image_dir, raw_image_dir, masks_out_dir)
                except Exception as e:
                    logger.warning(f"SKIP mask {mask_rel}: {e}")
                    skipped.append({"image": image_rel, "mask": mask_rel, "reason": str(e)})
                    continue

            if group_idx % 25 == 0:
                _write_summary(output_path, args.checkpoint, args.hit_threshold, records, skipped)
    finally:
        summary = _write_summary(output_path, args.checkpoint, args.hit_threshold, records, skipped)

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
