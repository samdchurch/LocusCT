#!/usr/bin/env python3
"""
Runs a fine-tuned VoxTell checkpoint (finetune_voxtell.py's model_state_dict
format) over the official (blind) ReXGroundingCT test set (official_splits/
ReXGroundingCT_test.json -- 582 findings across 300 unique volumes, every
entry has "mask": null) and saves one native-resolution prediction NIfTI per
volume to <run_dir>/official_predictions/, where <run_dir> is the
checkpoint's own run directory -- for submission to the ReXGroundingCT
leaderboard. VoxTell counterpart to predict_rexgroundingct_test.py (our own
Grounder model's version of this script); same output convention/location.

No resampling from a separate model grid: unlike Grounder, VoxTell runs
full-volume sliding-window inference directly on native-resolution images
(sliding_window_predict, imported from evaluate_finetuned_voxtell_ed.py) --
there's no resampled-grid model space to resample predictions back out of.
The manifest's "image" field still points at the resampled tree
("ReXGroundingCT/resampled/images/..."), used here only to group samples by
volume and recover each volume's name; native_image_path() resolves it to
the matching file inference actually runs on, under ReXGroundingCT/original/
(same relative path, "resampled/" swapped for "original/").

Predictions come out in load_nifti_canonical's (D, H, W) RAS+ layout, not
each native file's own on-disk orientation -- _canonical_mask_affine
(imported from evaluate_ed_official_test.py) recovers a correct affine for
that layout without hand-inverting the reorientation, and nibabel's
resample_from_to (nearest-neighbor) puts the prediction back onto a
reference native file's raw grid before saving -- same trick
predict_rexgroundingct_test.py uses, just with the native file standing in
for that script's separate resampled-grid image.

All findings for a volume share the exact same "image" value (one physical
CT per volume, not per finding -- the filename's trailing "_<digit>" is part
of the volume's own name, not a finding index).
Grouping samples by image (needed anyway: VoxTell batches every sentence for
one image into a single forward pass) puts a volume's whole finding set in
one inference call, and sliding_window_predict already returns one
probability map per sentence in that call -- so each finding's mask is kept
as its own separate slice (by position in the group, matching
MICCAI_challenge_dataset.json's "findings" order -- see
predict_rexgroundingct_test.py's _TestOnlyGrounderDataset docstring for how
that was verified) rather than OR'd into one undifferentiated mask.

Output filenames and per-volume finding count/shape are validated against
MICCAI_challenge_dataset.json (--challenge-json), same as
predict_rexgroundingct_test.py.

Usage
-----
    python predict_voxtell_rexgroundingct_test.py --checkpoint runs/voxtell_finetune/checkpoints/best.pt
    python predict_voxtell_rexgroundingct_test.py --checkpoint ... --multi-window
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
from nibabel.processing import resample_from_to
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import load_nifti_canonical
from evaluate_ed_official_test import _canonical_mask_affine
from evaluate_finetuned_voxtell_ed import sliding_window_predict
from finetune_voxtell import (
    DEFAULT_MODEL_DIR,
    DEFAULT_TEXT_ENCODER,
    apply_windows,
    build_voxtell_model,
    embed_sentences,
    load_text_backbone,
)

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "ReXGroundingCT_test.json"
# ReXGroundingCT lives directly under DATA_ROOT on our cluster, not a separate
# "public_datasets" mount -- same convention finetune_voxtell.py's own
# defaults and evaluate_finetuned_voxtell_rexgroundingct.py use.
DEFAULT_DATA_ROOT = Path("/path/to/data")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                         help="finetune_voxtell.py checkpoint (e.g. runs/voxtell_finetune/checkpoints/best.pt). "
                              "Output goes to this checkpoint's own run dir (parent of checkpoints/), "
                              "not a chosen --output-dir")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                         help="Base dir the manifest's relative image paths resolve against")
    parser.add_argument("--challenge-json", type=Path, default=None,
                         help="MICCAI_challenge_dataset.json, authoritative source for output filenames "
                              "and per-volume finding count/native shape (default: <data-root>/ReXGroundingCT/"
                              "MICCAI_challenge_dataset.json)")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                         help="VoxTell architecture/plans dir (plans.json) -- weights come from --checkpoint, not this")
    parser.add_argument("--text-encoder", default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--multi-window", dest="multi_window", action="store_true", default=False,
                         help="Must match how --checkpoint was fine-tuned (default: single Z-score-normalized channel)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def native_image_path(data_root: Path, resampled_image_rel: str) -> Path:
    """'ReXGroundingCT/resampled/images/...' -> 'ReXGroundingCT/original/images/...'"""
    native_rel = resampled_image_rel.replace("ReXGroundingCT/resampled/", "ReXGroundingCT/original/", 1)
    return data_root / native_rel


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    challenge_json_path = args.challenge_json or (args.data_root / "ReXGroundingCT" / "MICCAI_challenge_dataset.json")
    with open(challenge_json_path) as f:
        challenge_data = json.load(f)
    challenge_index: dict[str, dict] = {e["name"]: e for e in challenge_data["test"]}
    logger.info(f"Loaded {len(challenge_index)} test volume(s) from {challenge_json_path}")

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
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    logger.info(f"Loading frozen text backbone {args.text_encoder}")
    tokenizer, text_backbone = load_text_backbone(args.text_encoder, device)

    by_image: dict[str, list[dict]] = defaultdict(list)
    for s in kept:
        by_image[s["image"]].append(s)
    logger.info(f"Test images: {len(by_image)} ({len(kept)} sentences)")

    # pred_by_volume[volume_name][finding_idx]: canonical (D, H, W) uint8 mask, one per
    # sentence in that volume's group (in manifest order == MICCAI_challenge_dataset.json
    # "findings" order, see module docstring).
    # native_path_by_volume[volume_name]: the volume's one native file -- every finding in
    # a group comes from the same image, so there's exactly one per volume, both as the
    # source for _canonical_mask_affine and as the resample_from_to target grid.
    pred_by_volume: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    native_path_by_volume: dict[str, Path] = {}
    skipped: list[dict] = []

    for image_rel, group in tqdm(sorted(by_image.items()), desc="Running inference"):
        volume_name = Path(image_rel).name  # matches MICCAI_challenge_dataset.json's "name" field
        native_path = native_image_path(args.data_root, image_rel)
        if not native_path.exists():
            skipped.append({"image": image_rel, "reason": "native_image_not_found", "expected_path": str(native_path)})
            continue

        try:
            image_np = load_nifti_canonical(str(native_path))  # (D, H, W) raw HU, native resolution
            windowed = apply_windows(image_np, args.multi_window)
            image_t = torch.from_numpy(windowed).float()
            sentences = [s["sentence"] for s in group]
            text_embedding = embed_sentences(tokenizer, text_backbone, sentences, device)
            probs = sliding_window_predict(model, image_t, text_embedding, device)  # (N, D, H, W), one per sentence
        except Exception as e:
            logger.warning(f"SKIP image {image_rel}: {e}", exc_info=True)
            skipped.append({"image": image_rel, "reason": str(e)})
            continue

        for finding_idx, prob_map in enumerate(probs):
            pred_by_volume[volume_name][finding_idx] = (prob_map > args.threshold).astype(np.uint8)
        native_path_by_volume[volume_name] = native_path

    run_dir = Path(args.checkpoint).resolve().parent.parent
    out_dir = run_dir / "official_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing official predictions -> {out_dir}")

    n_written = 0
    for volume_name, finding_preds in tqdm(sorted(pred_by_volume.items()), desc="Resampling + saving"):
        try:
            challenge_entry = challenge_index.get(volume_name)
            if challenge_entry is None:
                skipped.append({"volume": volume_name, "reason": "not present in MICCAI_challenge_dataset.json test split"})
                continue

            expected_indices = list(range(len(challenge_entry["findings"])))
            finding_indices = sorted(finding_preds)
            if finding_indices != expected_indices:
                skipped.append({
                    "volume": volume_name,
                    "reason": f"finding indices {finding_indices} != expected {expected_indices} per MICCAI_challenge_dataset.json",
                })
                continue

            ref_native_img = nib.load(str(native_path_by_volume[volume_name]))
            affine = _canonical_mask_affine(ref_native_img)

            native_slices = []
            for idx in finding_indices:
                mask_img = nib.Nifti1Image(finding_preds[idx], affine)
                resampled = resample_from_to(mask_img, ref_native_img, order=0, mode="constant", cval=0)
                native_slices.append(np.asanyarray(resampled.dataobj).astype(np.uint8))
            pred_4d_native = np.stack(native_slices, axis=0)

            expected_shape = list(challenge_entry["shape"])
            if list(pred_4d_native.shape[1:]) != expected_shape:
                skipped.append({
                    "volume": volume_name,
                    "reason": f"native shape {list(pred_4d_native.shape[1:])} != MICCAI_challenge_dataset.json shape {expected_shape}",
                })
                continue

            nib.save(nib.Nifti1Image(pred_4d_native, ref_native_img.affine), str(out_dir / volume_name))
            n_written += 1
        except Exception as e:
            logger.warning(f"SKIP {volume_name}: {e}", exc_info=True)
            skipped.append({"volume": volume_name, "reason": str(e)})

    if skipped:
        with open(out_dir / "skipped.json", "w") as f:
            json.dump(skipped, f, indent=2)
        logger.warning(f"{len(skipped)} sample(s)/volume(s) skipped, see skipped.json")

    logger.info(f"Wrote {n_written}/{len(pred_by_volume)} volume(s) to {out_dir}")


if __name__ == "__main__":
    main()
