#!/usr/bin/env python3
"""
Scores OUR fine-tuned VoxTell checkpoint (finetune_voxtell.py's
model_state_dict format) on finetune_voxtell.py's own curated ED+ONC
validation split (official_splits/curated_ed_onc_val_data.json, i.e.
finetune_voxtell.DEFAULT_VAL_MANIFEST), with real full-volume
sliding-window inference -- see evaluate_finetuned_voxtell_ed.py's module
docstring for why this exists instead of the fast single-patch validation
proxy finetune_voxtell.py itself uses each epoch, and how the tiling works.
sliding_window_predict is imported directly from there.

This is in-domain val, not a held-out test set (finetune_voxtell.py trains
against this manifest's own Dice/loss each epoch) -- use it to sanity-check
a checkpoint or compare epochs, not as a generalization number; for that see
evaluate_finetuned_voxtell_ed.py/_onc.py (official held-out test sets) or
evaluate_finetuned_voxtell_rexgroundingct.py (external benchmark).

Curated val mask paths ("accession/mask_....nii.gz") don't embed a finding
category the way ED official test's ("CATEGORY/accession/Struct_....nii.gz")
does, so grouping uses each sample's "region" field from the manifest
instead (same convention evaluate_voxtell_rexgroundingct.py uses for this
manifest's "region": "Abdomen"/"Chest"/etc.). Reporting/visualization
helpers (_safe_filename, _save_figure, region_of) are reused directly from
there; _write_summary too, since it already groups into "by_region".

Usage
-----
    python evaluate_finetuned_voxtell_val.py --checkpoint runs/voxtell_finetune/checkpoints/best.pt
    python evaluate_finetuned_voxtell_val.py --checkpoint ... --visualize
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
from evaluate_finetuned_voxtell_ed import sliding_window_predict
from evaluate_voxtell_rexgroundingct import _safe_filename, _save_figure, _write_summary, region_of
from finetune_voxtell import (
    DEFAULT_MASK_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_TEXT_ENCODER,
    DEFAULT_VAL_MANIFEST,
    apply_windows,
    build_voxtell_model,
    embed_sentences,
    load_text_backbone,
)
from utils.metrics import dice_score

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / DEFAULT_VAL_MANIFEST
# Native (pre-resample) roots, same convention as finetune_voxtell.py -- the
# curated val split's masks live in the same native labels/ tree as the main
# dataset (same as ONC official test's, unlike ED's separate ED_TEST_SET/).
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
    parser.add_argument("--tile-overlap", type=float, default=0.0,
                         help="Fraction (0.0-<1.0) of patch_size neighboring sliding-window tiles overlap by, "
                              "Gaussian-blended (nnU-Net-style) -- see sliding_window_predict's docstring in "
                              "evaluate_finetuned_voxtell_ed.py")
    parser.add_argument("--output", default="outputs/eval/voxtell_finetuned_val/results.json")
    parser.add_argument("--hit-threshold", type=float, default=0.1,
                         help="Dice value at/above which a sample counts as a 'hit'")
    parser.add_argument("--visualize", dest="visualize", action="store_true", default=False,
                         help="Write per-case GT/pred overlay PNGs (off by default)")
    parser.add_argument("--viz-output-dir", default=None,
                         help="Where to write visualization PNGs. Default: <output's parent>/viz")
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Evaluate only the first N cases (manifest order) for a quick run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    with open(args.manifest) as f:
        samples = json.load(f)
    kept = [s for s in samples if s.get("sentence")]
    n_dropped = len(samples) - len(kept)
    if n_dropped:
        logger.warning(f"Dropping {n_dropped}/{len(samples)} sample(s) with no sentence: "
                        f"{[s['mask'] for s in samples if not s.get('sentence')]}")

    if args.max_samples is not None:
        kept = kept[:args.max_samples]
        logger.info(f"--max-samples set: evaluating only the first {len(kept)} case(s)")

    region_lookup = {s["mask"]: region_of(s) for s in kept}

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

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)

    by_image: dict[str, list[dict]] = defaultdict(list)
    for s in kept:
        by_image[s["image"]].append(s)

    records: list[dict] = []
    skipped: list[dict] = []

    try:
        for group_idx, (image_rel, group) in enumerate(
            tqdm(sorted(by_image.items()), desc="Evaluating curated ED+ONC val set (fine-tuned VoxTell)")
        ):
            try:
                image_np = load_nifti_canonical(str(image_dir / image_rel))
                windowed = apply_windows(image_np, args.multi_window)
                image_t = torch.from_numpy(windowed).float()
                sentences = [s["sentence"] for s in group]
                text_embedding = embed_sentences(tokenizer, text_backbone, sentences, device)
                probs = sliding_window_predict(model, image_t, text_embedding, device,
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

                    reg = region_lookup.get(mask_rel, "UNKNOWN")
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
