#!/usr/bin/env python3
"""
Scores OUR fine-tuned VoxTell checkpoint (finetune_voxtell.py's
model_state_dict format) on the ReXGroundingCT val split
(official_splits/ReXGroundingCT_val.json) -- an external public benchmark,
NOT part of finetune_voxtell.py's own training data, so this measures
cross-dataset generalization after in-house fine-tuning, not in-domain
performance. Real full-volume sliding-window inference -- see
evaluate_finetuned_voxtell_ed.py's module docstring for why this exists
instead of reusing evaluate_voxtell_rexgroundingct.py's VoxTellPredictor
path and how the tiling works. sliding_window_predict is imported directly
from there.

Mask handling (stem/finding-index parsing, native path resolution, 4D
segmentation splitting via a borrowed image affine) reuses
evaluate_voxtell_rexgroundingct.py's helpers directly -- see that module's
docstring for the orientation caveat it flags (not empirically validated).
The extracted finding mask is put through the SAME as_closest_canonical ->
transpose(2,1,0) -> _normalize_depth_axis pipeline load_nifti_canonical
applies to every other volume (reusing data.dataset._normalize_depth_axis
directly, see _load_canonical_finding_mask below), so it ends up in the
identical (D, H, W) convention as the model's image input --
evaluate_voxtell_rexgroundingct.py's own version doesn't need this since
VoxTellPredictor/NibabelIOWithReorient uses a different axis convention
than our model was trained on.

Usage
-----
    python evaluate_finetuned_voxtell_rexgroundingct.py --checkpoint runs/voxtell_finetune/checkpoints/best.pt
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import _normalize_depth_axis, load_nifti_canonical
from evaluate_finetuned_voxtell_ed import sliding_window_predict
from evaluate_voxtell_rexgroundingct import (
    _finding_axis,
    _safe_filename,
    _save_figure,
    _write_summary,
    native_image_path,
    native_seg_path,
    parse_stem_and_finding,
    region_of,
)
from finetune_voxtell import (
    DEFAULT_MODEL_DIR,
    DEFAULT_TEXT_ENCODER,
    apply_windows,
    build_voxtell_model,
    embed_sentences,
    load_text_backbone,
)
from utils.metrics import dice_score

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "ReXGroundingCT_val.json"
# ReXGroundingCT lives directly under DATA_ROOT on our cluster, not a
# separate "public_datasets" mount (see submit_rexgroundingct_eval.sh).
DEFAULT_DATA_ROOT = "/path/to/data"

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
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                         help="Base dir the native 'ReXGroundingCT/original/images|segmentations/...' paths resolve against")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                         help="VoxTell architecture/plans dir (plans.json) -- weights come from --checkpoint, not this")
    parser.add_argument("--text-encoder", default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--multi-window", dest="multi_window", action="store_true", default=False,
                         help="Must match how --checkpoint was fine-tuned (default: single Z-score-normalized channel)")
    parser.add_argument("--tile-overlap", type=float, default=0.0,
                         help="Fraction (0.0-<1.0) of patch_size neighboring sliding-window tiles overlap by, "
                              "Gaussian-blended (nnU-Net-style) -- see sliding_window_predict's docstring in "
                              "evaluate_finetuned_voxtell_ed.py")
    parser.add_argument("--output", default="outputs/eval/voxtell_finetuned_rex/results.json")
    parser.add_argument("--hit-threshold", type=float, default=0.1,
                         help="Dice value at/above which a sample counts as a 'hit'")
    parser.add_argument("--visualize", dest="visualize", action="store_true", default=False,
                         help="Write per-case GT/pred overlay PNGs (off by default)")
    parser.add_argument("--viz-output-dir", default=None,
                         help="Where to write visualization PNGs. Default: <output's parent>/viz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _load_canonical_finding_mask(finding_arr: np.ndarray, img_affine: np.ndarray, seg_path: str) -> np.ndarray:
    """Same as_closest_canonical -> transpose(2,1,0) -> _normalize_depth_axis
    pipeline load_nifti_canonical applies when loading a file from disk,
    applied here to an in-memory finding slice (borrowed image affine,
    since the segmentation's own affine isn't trustworthy -- see module
    docstring) so it matches the model's image input's (D, H, W) convention."""
    finding_nib = nib.Nifti1Image(finding_arr.astype(np.uint8), affine=img_affine)
    canonical = nib.as_closest_canonical(finding_nib)
    arr = canonical.get_fdata(dtype=np.float32).transpose(2, 1, 0)
    return _normalize_depth_axis(arr, seg_path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    with open(args.manifest) as f:
        samples = json.load(f)
    kept = [s for s in samples if s.get("sentence")]
    n_dropped = len(samples) - len(kept)
    if n_dropped:
        logger.warning(f"Dropping {n_dropped}/{len(samples)} sample(s) with no sentence")

    num_channels = 3 if args.multi_window else 1
    logger.info(f"Building VoxTell model (input_channels={num_channels}) from {args.model_dir}")
    model = build_voxtell_model(Path(args.model_dir), num_channels).to(device)

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

    by_stem: dict[str, list[dict]] = defaultdict(list)
    for s in kept:
        stem, finding_idx = parse_stem_and_finding(s["mask"])
        by_stem[stem].append({**s, "_finding_idx": finding_idx})

    records: list[dict] = []
    skipped: list[dict] = []

    try:
        for group_idx, (stem, group) in enumerate(
            tqdm(sorted(by_stem.items()), desc="Evaluating ReXGroundingCT val (fine-tuned VoxTell)")
        ):
            group = sorted(group, key=lambda s: s["_finding_idx"])
            img_path = native_image_path(args.data_root, group[0]["image"])
            seg_path = native_seg_path(args.data_root, stem)

            try:
                image_np = load_nifti_canonical(img_path)
                windowed = apply_windows(image_np, args.multi_window)
                image_t = torch.from_numpy(windowed).float()
                sentences = [s["sentence"] for s in group]
                text_embedding = embed_sentences(tokenizer, text_backbone, sentences, device)
                probs = sliding_window_predict(model, image_t, text_embedding, device,
                                                tile_overlap=args.tile_overlap)

                raw_img_nib = nib.load(img_path)
                raw_seg_nib = nib.load(seg_path)
                raw_seg_data = raw_seg_nib.get_fdata()
                finding_axis = _finding_axis(raw_seg_data.shape, raw_img_nib.shape)
            except Exception as e:
                logger.warning(f"SKIP volume {stem}: {e}")
                skipped.extend({"stem": stem, "mask": s["mask"], "reason": str(e)} for s in group)
                continue

            for i, s in enumerate(group):
                mask_rel = s["mask"]
                try:
                    finding_idx = s["_finding_idx"]
                    finding_arr = (
                        raw_seg_data[finding_idx] if finding_axis == 0
                        else raw_seg_data[..., finding_idx]
                    )
                    gt = (_load_canonical_finding_mask(finding_arr, raw_img_nib.affine, seg_path) > 0.5).astype(np.float32)
                    pr = probs[i].astype(np.float32)
                    if gt.shape != pr.shape:
                        raise ValueError(f"image/mask shape mismatch: pred={pr.shape} gt={gt.shape}")

                    dice = dice_score(
                        torch.from_numpy(pr[None]), torch.from_numpy(gt[None]),
                        threshold=0.5, smooth=1.0, from_logits=False,
                    )[0].item()

                    reg = region_of(s)
                    records.append({
                        "id": mask_rel,
                        "region": reg,
                        "dice": dice,
                        "hit": bool(dice >= args.hit_threshold),
                    })

                    if args.visualize:
                        reg_dir = viz_dir / reg
                        reg_dir.mkdir(parents=True, exist_ok=True)
                        _save_figure(
                            image_np, gt, pr, s["sentence"], dice,
                            reg_dir / f"{_safe_filename(mask_rel)}.png",
                        )
                except Exception as e:
                    logger.warning(f"SKIP mask {mask_rel}: {e}")
                    skipped.append({"stem": stem, "mask": mask_rel, "reason": str(e)})
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
        logger.info("  By region:")
        for reg, s in summary["by_region"].items():
            logger.info(
                f"    {reg:<15} n={s['n_samples']:<4} "
                f"Dice={s['dice_mean']:.4f}+/-{s['dice_std']:.4f}  Hit={s['hit_rate']:.4f}"
            )
        if args.visualize:
            logger.info(f"Wrote {len(records)} visualization(s) -> {viz_dir}")


if __name__ == "__main__":
    main()
