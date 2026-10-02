#!/usr/bin/env python3
"""
Runs a fine-tuned VoxTell checkpoint (finetune_voxtell.py's model_state_dict
format) over the Merlin abdominal CT dataset's per-study atomic findings --
VoxTell counterpart to predict_merlin.py (our own Grounder model's version
of this script). Merlin data loading (merlin_sentences.json format,
--status/--include-normal/--n/--seed sampling, --category-findings-dir mode)
is reused directly from there, nothing duplicated -- see that script's
module docstring for the full data format/sampling details. There's no
ground truth here, just raw inference over Merlin's own report-derived text
(or, with --category-findings-dir, ED-category phrase text); see
visualize_merlin_predictions.py to look at the results.

Unlike predict_merlin.py's placeholder identity-affine mask dump, predicted
masks here are saved with a REAL affine via _save_predicted_mask (imported
from evaluate_finetuned_voxtell_ed.py, itself reusing evaluate_ed_official_
test.py's _canonical_mask_affine) -- so they are spatially registered and
line up against a fresh load of their source image in any NIfTI viewer, not
just index-for-index with a from-scratch load_nifti_canonical call. This is
a strict improvement, not a behavior change visualize_merlin_predictions.py
needs to know about -- it reads mask arrays via plain nib.load(...).get_fdata()
and never touches the affine.

Full-volume sliding-window inference (sliding_window_predict, imported from
evaluate_finetuned_voxtell_ed.py) is used instead of Grounder's resize-to-
fixed-grid approach, same as predict_voxtell_rexgroundingct_test.py. Always
uses live text encoding (load_text_backbone/embed_sentences) -- unlike
predict_merlin.py, there's no --embedding-cache option here, matching every
other finetune_voxtell.py eval script's convention.

Usage
-----
    python predict_finetuned_voxtell_merlin.py --checkpoint runs/voxtell_finetune/checkpoints/best.pt
    python predict_finetuned_voxtell_merlin.py --checkpoint ... --n 0  # run on everything
    python predict_finetuned_voxtell_merlin.py --checkpoint ... \
        --category-findings-dir merlin_ed_category_findings --output-dir outputs/eval/merlin_ed_categories
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import load_nifti_canonical
from evaluate_finetuned_voxtell_ed import _save_predicted_mask, sliding_window_predict
from finetune_voxtell import (
    DEFAULT_MODEL_DIR,
    DEFAULT_TEXT_ENCODER,
    apply_windows,
    build_voxtell_model,
    embed_sentences,
    load_text_backbone,
)
from predict_merlin import (
    DEFAULT_SENTENCES,
    load_ed_category_samples,
    load_merlin_samples,
    sample_key,
)

DEFAULT_IMAGE_DIR = Path(
    "/path/to/data/public_datasets/merlinabdominalctdataset/merlin_data_resampled"
)

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
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/voxtell_merlin_predictions"))
    parser.add_argument("--status", nargs="+", default=["present"],
                         help="Only run inference on atomic_findings with one of these status values (default: present)")
    parser.add_argument("--include-normal", action="store_true",
                         help="Include findings that just say normal/unremarkable/patent (see predict_merlin.py's "
                              "is_normal_finding) -- skipped by default since there's nothing to localize for those")
    parser.add_argument("--n", type=int, default=20,
                         help="Randomly sample this many (study, finding) pairs to run inference on (default: 20; "
                              "0 = everything). Ignored entirely when --category-findings-dir is given.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--category-findings-dir", type=Path, default=None,
        help="Directory of per-ED-category *.json files (see predict_merlin.py's own --category-findings-dir "
             "docs for the format and why there's no merlin_sentences.json cross-reference). Overrides "
             "--n/--seed/--sentences/--status/--include-normal.",
    )
    parser.add_argument(
        "--max-per-category", type=int, default=50,
        help="Cap each ed-category to at most this many findings (random subset, seeded by --seed) -- only "
             "applies with --category-findings-dir. 0 = no cap.",
    )
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
                         help="See sliding_window_predict's docstring in evaluate_finetuned_voxtell_ed.py")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--raw-image-dir", default=None,
                         help="If given, predicted masks are additionally resampled onto each study's original "
                              "raw grid instead of being saved directly on --image-dir's grid (see "
                              "_save_predicted_mask) -- only meaningful if a separate raw Merlin tree exists "
                              "on the machine this runs on.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    device = torch.device(args.device)
    if device.type == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES"):
        # --device's own default (parse_args) silently falls back to "cpu" whenever
        # torch.cuda.is_available() is False -- including when a GPU WAS allocated
        # (CUDA_VISIBLE_DEVICES set by Slurm/singularity --nv) but torch's CUDA init
        # itself failed on this node (e.g. a "CUDA initialization: CUDA unknown
        # error" UserWarning at startup). Sliding-window inference on CPU is ~75x
        # slower than on an H200 -- catching this here avoids silently burning a
        # multi-hour GPU allocation running on the CPU instead, undetected until
        # someone happens to check the logs mid-run.
        raise RuntimeError(
            f"torch.cuda.is_available() is False despite CUDA_VISIBLE_DEVICES="
            f"{os.environ['CUDA_VISIBLE_DEVICES']!r} (this job was allocated a GPU) -- torch's CUDA "
            f"init likely failed on this node. Resubmit (Slurm will likely place it on a different "
            f"node) rather than let this run on CPU for hours; pass --device cpu explicitly if CPU "
            f"inference is genuinely intended."
        )
    raw_image_dir = Path(args.raw_image_dir) if args.raw_image_dir else None

    sample_categories: dict[str, set[str]] | None = None  # sample_key -> {category names}
    if args.category_findings_dir:
        samples, sample_categories = load_ed_category_samples(
            args.category_findings_dir, args.image_dir,
            max_per_category=args.max_per_category or None, seed=args.seed,
        )
    else:
        samples = load_merlin_samples(args.sentences, args.image_dir, args.status, skip_normal=not args.include_normal)
        if args.n:
            rng = random.Random(args.seed)
            samples = rng.sample(samples, min(args.n, len(samples)))
            logger.info(f"Randomly sampled {len(samples)} sample(s) for inference (seed={args.seed}) "
                        f"-- matches what visualize_merlin_predictions.py will show")
    if not samples:
        logger.error("No samples to run -- check --sentences/--image-dir/--status/--category-findings-dir")
        sys.exit(1)

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
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    logger.info(f"Loading frozen text backbone {args.text_encoder}")
    tokenizer, text_backbone = load_text_backbone(args.text_encoder, device)

    # Group by study so each volume is loaded and inferred on once, with all its
    # findings batched into a single sliding_window_predict call -- same reasoning as
    # every other finetune_voxtell.py eval script.
    by_study: dict[str, list[dict]] = {}
    for s in samples:
        by_study.setdefault(s["study_id"], []).append(s)
    logger.info(f"Studies: {len(by_study)} ({len(samples)} findings)")

    if sample_categories is not None:
        category_names = sorted({c for cats in sample_categories.values() for c in cats})
        masks_dirs = {c: args.output_dir / c / "predicted_masks" for c in category_names}
        for d in masks_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        results_by_category: dict[str, list[dict]] = {c: [] for c in category_names}
    else:
        masks_dir = args.output_dir / "predicted_masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict] = []

    skipped: list[dict] = []

    for study_id, group in tqdm(sorted(by_study.items()), desc="Running inference"):
        image_rel = f"{study_id}.nii.gz"
        try:
            image_np = load_nifti_canonical(str(args.image_dir / image_rel))
            windowed = apply_windows(image_np, args.multi_window)
            image_t = torch.from_numpy(windowed).float()
            sentences = [s["sentence"] for s in group]
            text_embedding = embed_sentences(tokenizer, text_backbone, sentences, device)
            probs = sliding_window_predict(model, image_t, text_embedding, device,
                                            patch_size=patch_size, tile_overlap=args.tile_overlap)  # (N, D, H, W)
        except Exception as e:
            logger.warning(f"SKIP study {study_id}: {e}", exc_info=True)
            skipped.extend({"study_id": study_id, "finding_idx": s["finding_idx"], "reason": str(e)} for s in group)
            continue

        for i, s in enumerate(group):
            finding_idx = s["finding_idx"]
            sk = sample_key(study_id, finding_idx)
            pr = probs[i].astype(np.float32)
            record_base = {
                "study_id": study_id,
                "finding_idx": finding_idx,
                "sentence": s["sentence"],
                "organ": s["organ"],
                "laterality": s["laterality"],
                "image_path": str(args.image_dir / image_rel),
                "voxel_count": int((pr > args.threshold).sum()),
            }

            if sample_categories is not None:
                # Inference ran once for this sample even if it matches multiple
                # categories -- save the same prediction into each matching folder.
                for category in sorted(sample_categories[sk]):
                    _save_predicted_mask(pr, args.threshold, sk, image_rel, args.image_dir,
                                          raw_image_dir, masks_dirs[category])
                    mask_path = masks_dirs[category] / f"{sk}_pred.nii.gz"
                    results_by_category[category].append({**record_base, "mask_path": str(mask_path)})
            else:
                _save_predicted_mask(pr, args.threshold, sk, image_rel, args.image_dir, raw_image_dir, masks_dir)
                mask_path = masks_dir / f"{sk}_pred.nii.gz"
                results.append({**record_base, "mask_path": str(mask_path)})

    if sample_categories is not None:
        total = 0
        for category, cat_results in results_by_category.items():
            predictions_path = args.output_dir / category / "predictions.json"
            with open(predictions_path, "w") as f:
                json.dump(cat_results, f, indent=2)
            n_empty = sum(1 for r in cat_results if r["voxel_count"] == 0)
            logger.info(f"[{category}] Wrote {len(cat_results)} prediction(s) -> {predictions_path} "
                        f"({n_empty} empty)")
            total += len(cat_results)
        logger.info(f"Wrote {total} total prediction(s) across {len(results_by_category)} "
                    f"category folder(s) under {args.output_dir}")
    else:
        predictions_path = args.output_dir / "predictions.json"
        with open(predictions_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Wrote {len(results)} prediction(s) -> {predictions_path}")
        logger.info(f"  {sum(1 for r in results if r['voxel_count'] == 0)}/{len(results)} predicted an empty mask")

    if skipped:
        with open(args.output_dir / "skipped.json", "w") as f:
            json.dump(skipped, f, indent=2)
        logger.warning(f"{len(skipped)} sample(s) skipped, see skipped.json")


if __name__ == "__main__":
    main()
