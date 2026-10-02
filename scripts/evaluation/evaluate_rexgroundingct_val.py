#!/usr/bin/env python3
"""
Score a Grounder checkpoint on the ReXGroundingCT val split (official_splits/
ReXGroundingCT_val.json) using the official ReXrank scoring metric, so numbers are
directly comparable to the challenge leaderboard.

The MICCAI challenge's own evaluator (rexrank_eval.py, shipped alongside the dataset)
operates on one file per *volume*, each a 4D NIfTI stacking every finding's mask -- not
the per-(image, finding) records our manifest/dataloader use. This script therefore:
  1. Runs the model over every (image, sentence) sample in the val manifest.
  2. Groups predictions by source volume (parsed from each sample's mask filename,
     "<volume_stem>_<finding_idx>.nii.gz") and stacks them into one 4D array per volume,
     in finding-index order.
  3. Builds the matching GT 4D stack by re-reading the individual per-finding mask files
     already produced by resample_rexgroundingct.py through the same RAS+
     canonicalization the dataset applies (load_nifti_canonical), so GT lines up
     voxel-for-voxel with the model's own (D, H, W) prediction space instead of each
     file's raw on-disk orientation.
  4. Writes gt/, pred/, and a dataset_json (the {"test": [{"seg_path": ...}, ...]}
     shape rexrank_eval.py expects -- the key is literally "test" regardless of which
     split we're scoring, that's just the field name their generic harness reads).
  5. Invokes rexrank_eval.py as a subprocess to produce the official summary metrics.

Evaluation happens in the resampled (352, 352, 180) grid -- the same space the model
was trained on -- not each scan's native resolution, so these numbers are an
apples-to-apples comparison across our own checkpoints but won't exactly match a
native-resolution leaderboard score.

Usage
-----
    python evaluate_rexgroundingct_val.py --config configs/default.yaml --checkpoint runs/default/checkpoints/best.pt
    python evaluate_rexgroundingct_val.py --config configs/default.yaml --checkpoint best.pt --output-dir /tmp/rex_val_eval
"""

import argparse
import copy
import json
import logging
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import build_dataloader, load_nifti_canonical
from models.grounder import Grounder
from train import apply_overrides
from training.trainer import _trim_padding

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "ReXGroundingCT_val.json"
DEFAULT_DATA_ROOT = Path("/path/to/data/public_datasets")
DEFAULT_REXRANK_EVAL_SCRIPT = DEFAULT_DATA_ROOT / "ReXGroundingCT" / "rexrank_eval.py"

# Mask filenames are "<volume_stem>_<finding_idx>.nii.gz" (resample_rexgroundingct.py);
# stem itself may end in digits (e.g. "..._a_1"), so greedily match everything before
# the LAST "_<digits>" as the stem.
STEM_FINDING_RE = re.compile(r"^(?P<stem>.+)_(?P<idx>\d+)$")

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
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                         help="Base dir the manifest's relative image/mask paths resolve against")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/rexgroundingct_val"))
    parser.add_argument("--rexrank-eval-script", type=Path, default=DEFAULT_REXRANK_EVAL_SCRIPT)
    parser.add_argument("--num-eval-workers", type=int, default=8,
                         help="Worker processes for rexrank_eval.py's own scoring pass")
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--cc-connectivity", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--global-only", action="store_true",
                         help="Only compute global Dice/hit-rate (skip instance-level matching)")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Dot-notation config overrides, e.g. model.text_encoder_name=/path/to/model",
    )
    return parser.parse_args()


def strip_nii_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return name


def parse_stem_and_finding(mask_rel_path: str) -> tuple[str, int]:
    stem_with_idx = strip_nii_ext(Path(mask_rel_path).name)
    m = STEM_FINDING_RE.match(stem_with_idx)
    if not m:
        raise ValueError(f"Can't parse volume stem/finding index from mask path: {mask_rel_path}")
    return m["stem"], int(m["idx"])


def main() -> None:
    args = parse_args()

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

    cfg_rex = copy.deepcopy(cfg)
    cfg_rex["data"]["image_dir"] = str(args.data_root)
    cfg_rex["data"]["mask_dir"] = str(args.data_root)
    loader = build_dataloader(str(args.manifest), cfg_rex, split="val", num_workers=0)

    segmentations_root = args.data_root / "ReXGroundingCT" / "resampled" / "segmentations"

    pred_by_volume: dict[str, dict[int, np.ndarray]] = defaultdict(dict)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference"):
            image = batch["image"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if use_cached_text:
                    text_feats = batch["text_feats"].to(device)
                    text_padding_mask = batch["text_padding_mask"].to(device)
                    logits = model(image, text_feats=text_feats, text_padding_mask=text_padding_mask)
                else:
                    input_ids = batch["input_ids"].to(device)
                    attn_mask = batch["attention_mask"].to(device)
                    logits = model(image, input_ids, attn_mask)
            logits, _ = _trim_padding(logits, batch["mask"], batch["pad_amounts"])
            probs = torch.sigmoid(logits).cpu().float().numpy()

            for i, mask_id in enumerate(batch["id"]):
                stem, finding_idx = parse_stem_and_finding(mask_id)
                pred_np = (probs[i, 0] > threshold).astype(np.uint8)  # (D, H, W), canonical RAS+
                pred_by_volume[stem][finding_idx] = pred_np

    output_dir = args.output_dir
    gt_dir = output_dir / "gt"
    pred_dir = output_dir / "pred"
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    dataset_entries = []
    skipped: list[dict] = []
    for stem, finding_preds in tqdm(sorted(pred_by_volume.items()), desc="Stacking per-volume NIfTIs"):
        finding_indices = sorted(finding_preds)
        gt_stack, pred_stack = [], []
        for idx in finding_indices:
            gt_path = segmentations_root / f"{stem}_{idx}.nii.gz"
            if not gt_path.exists():
                skipped.append({"stem": stem, "finding_idx": idx, "reason": "gt_not_found",
                                 "expected_path": str(gt_path)})
                gt_stack = None
                break
            # Load GT through the same RAS+ canonicalization the dataset applies to every
            # training/inference sample, so it lines up with pred_np's (D, H, W) space
            # voxel-for-voxel instead of relying on inverting as_closest_canonical by hand
            # (which can involve axis flips a plain transpose can't undo).
            gt_stack.append(load_nifti_canonical(str(gt_path)).astype(np.uint8))
            pred_stack.append(finding_preds[idx])

        if not gt_stack:
            continue

        gt_4d = np.stack(gt_stack, axis=0)
        pred_4d = np.stack(pred_stack, axis=0)
        if gt_4d.shape != pred_4d.shape:
            skipped.append({"stem": stem, "reason": "shape_mismatch",
                             "gt_shape": list(gt_4d.shape), "pred_shape": list(pred_4d.shape)})
            continue

        fname = f"{stem}.nii.gz"
        nib.save(nib.Nifti1Image(gt_4d, affine=np.eye(4)), str(gt_dir / fname))
        nib.save(nib.Nifti1Image(pred_4d, affine=np.eye(4)), str(pred_dir / fname))
        dataset_entries.append({"seg_path": fname})

    if skipped:
        logger.warning("%d volume(s) skipped, see skipped.json", len(skipped))
        with open(output_dir / "skipped.json", "w") as f:
            json.dump(skipped, f, indent=2)

    dataset_json_path = output_dir / "rexrank_dataset.json"
    with open(dataset_json_path, "w") as f:
        json.dump({"test": dataset_entries}, f, indent=2)
    logger.info(f"Wrote {len(dataset_entries)} volume(s) to score -> {dataset_json_path}")

    results_path = output_dir / "rexrank_results.json"
    cmd = [
        sys.executable, str(args.rexrank_eval_script),
        "--gt_dir", str(gt_dir),
        "--pred_dir", str(pred_dir),
        "--dataset_json", str(dataset_json_path),
        "--output_json", str(results_path),
        "--num_workers", str(args.num_eval_workers),
        "--min_size", str(args.min_size),
        "--cc_connectivity", str(args.cc_connectivity),
    ]
    if args.global_only:
        cmd.append("--global_only")
    logger.info("Running rexrank_eval.py: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    with open(results_path) as f:
        summary = json.load(f).get("summary", {})
    logger.info("ReXGroundingCT val results:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
