#!/usr/bin/env python3
"""
Scores OUR fine-tuned VoxTell checkpoint (finetune_voxtell.py's
model_state_dict format) on the official oncology held-out test set
(official_splits/onc_official_test_data.json), with real full-volume
sliding-window inference -- see evaluate_finetuned_voxtell_ed.py's module
docstring for why this exists instead of reusing evaluate_voxtell_onc.py's
VoxTellPredictor path (checkpoint format + preprocessing mismatch) and how
the tiling works. sliding_window_predict is imported directly from there.

Oncology mask paths ("accession/mask_....nii.gz") don't embed a finding
category the way ED's ("CATEGORY/accession/Struct_....nii.gz") do, so
grouping uses each sample's "finding" field from the manifest instead (same
convention evaluate_voxtell_onc.py uses). Its own reporting/visualization
helpers (_safe_filename, _save_figure, _write_summary) are reused directly.

Pass --save-masks (optionally with --raw-image-dir) to write predicted masks
as .nii.gz -- see evaluate_finetuned_voxtell_ed.py's module docstring for how
this works; _save_predicted_mask is imported directly from there.

Usage
-----
    python evaluate_finetuned_voxtell_onc.py --checkpoint runs/voxtell_finetune/checkpoints/best.pt
    python evaluate_finetuned_voxtell_onc.py --checkpoint ... --save-masks \
        --raw-image-dir /path/to/data/inhouse_abdominal_ct/nifti
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import load_nifti_canonical
from evaluate_finetuned_voxtell_ed import _save_predicted_mask, sliding_window_predict
from evaluate_voxtell_onc import _safe_filename, _save_figure, _write_summary
from finetune_voxtell import (
    DEFAULT_MASK_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_TEXT_ENCODER,
    apply_windows,
    build_voxtell_model,
    embed_sentences,
    load_text_backbone,
)
from utils.metrics import dice_score

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "onc_official_test_data.json"
# Native (pre-resample) roots, same convention as finetune_voxtell.py --
# oncology test masks live in the same native labels/ tree as the main
# dataset (unlike ED's separate ED_TEST_SET/), hence reusing
# finetune_voxtell.DEFAULT_MASK_DIR directly below.
DEFAULT_IMAGE_DIR = "/path/to/data/inhouse_abdominal_ct/nifti"

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
                              "Gaussian-blended (nnU-Net-style) -- see sliding_window_predict's docstring in "
                              "evaluate_finetuned_voxtell_ed.py")
    parser.add_argument("--output", default="outputs/eval/voxtell_finetuned_onc/results.json")
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

    finding_lookup = {s["mask"]: (s.get("finding") or "Unknown") for s in kept}

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
            tqdm(sorted(by_image.items()), desc="Evaluating oncology official test set (fine-tuned VoxTell)")
        ):
            try:
                image_np = load_nifti_canonical(str(image_dir / image_rel))
                windowed = apply_windows(image_np, args.multi_window)
                image_t = torch.from_numpy(windowed).float()
                sentences = [s["sentence"] for s in group]
                text_embedding = embed_sentences(tokenizer, text_backbone, sentences, device)
                probs = sliding_window_predict(model, image_t, text_embedding, device,
                                                patch_size=args.patch_size,
                                                tile_overlap=args.tile_overlap)
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

                    cat = finding_lookup.get(mask_rel, "Unknown")
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
        logger.info("  By finding:")
        for cat, s in summary["by_category"].items():
            logger.info(
                f"    {cat:<15} n={s['n_samples']:<4} "
                f"Dice={s['dice_mean']:.4f}+/-{s['dice_std']:.4f}  Hit={s['hit_rate']:.4f}"
            )
        if args.visualize:
            logger.info(f"Wrote {len(records)} visualization(s) -> {viz_dir}")


if __name__ == "__main__":
    main()
