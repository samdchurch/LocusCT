import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from data.dataset import build_dataloader
from models.grounder import Grounder
from training.trainer import _trim_padding
from utils.metrics import dice_score, iou_score, precision_recall

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3D Visual Grounder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument(
        "--save_masks",
        action="store_true",
        help="Save predicted masks as .nii.gz (requires nibabel)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = cfg["inference"].get("threshold", 0.5)

    # Same reasoning as evaluate_rexgroundingct_val.py / evaluate_ed_official_test.py:
    # a checkpoint trained with embedding_cache set has no text_encoder.* weights at
    # all (train.py never loads the backbone), so this must mirror train.py's flag
    # exactly or model.load_state_dict() below fails on missing keys.
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

    manifest_key = f"{args.split}_manifest"
    # Match train.py's seed so a max_samples subset (if set) is the exact same
    # sample of the manifest a training run built its loaders from, rather than
    # silently defaulting to build_dataloader's own seed=42.
    loader = build_dataloader(cfg["data"][manifest_key], cfg, split=args.split, num_workers=0,
                               seed=cfg.get("seed", 42))

    per_sample: list[dict] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
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
                per_sample.append(
                    {
                        "id": sample_id,
                        "dice": dices[i].item(),
                        "iou": ious[i].item(),
                        "precision": prec[i].item(),
                        "recall": rec[i].item(),
                    }
                )

            if args.save_masks:
                _save_masks(logits, batch, threshold, args.output)

    dice_vals = [s["dice"] for s in per_sample]
    iou_vals = [s["iou"] for s in per_sample]
    prec_vals = [s["precision"] for s in per_sample]
    rec_vals = [s["recall"] for s in per_sample]

    summary = {
        "split": args.split,
        "n_samples": len(per_sample),
        "dice_mean": float(np.mean(dice_vals)),
        "dice_std": float(np.std(dice_vals)),
        "iou_mean": float(np.mean(iou_vals)),
        "iou_std": float(np.std(iou_vals)),
        "precision_mean": float(np.mean(prec_vals)),
        "recall_mean": float(np.mean(rec_vals)),
        "hit_rate": float(np.mean([d >= 0.1 for d in dice_vals])),
        "per_sample": per_sample,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Results → {args.output}\n"
        f"  Dice:  {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}\n"
        f"  IoU:   {summary['iou_mean']:.4f} ± {summary['iou_std']:.4f}\n"
        f"  Prec:  {summary['precision_mean']:.4f}  Rec: {summary['recall_mean']:.4f}\n"
        f"  Hit:   {summary['hit_rate']:.4f}"
    )


def _save_masks(
    logits: torch.Tensor,
    batch: dict,
    threshold: float,
    output_path: str,
) -> None:
    import nibabel as nib

    out_dir = Path(output_path).parent / "predicted_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = torch.sigmoid(logits).cpu().float().numpy()
    for i, sample_id in enumerate(batch["id"]):
        # Save in the same canonical (D, H, W) order load_nifti_canonical produces --
        # transposing back to "on-disk order" only correctly inverts
        # nib.as_closest_canonical() when that step was a pure permutation; for any
        # file where it also flipped an axis (e.g. LPS stored CT -> RAS+), a plain
        # transpose(2, 1, 0) silently mirrors the saved mask (see evaluate_rexgroundingct_val.py's
        # history for this exact bug). affine is a placeholder, not the source file's
        # real affine, so this won't spatially overlay a viewer's copy of the original
        # scan -- use visualize.py for that.
        mask_np = (probs[i, 0] > threshold).astype(np.uint8)
        img = nib.Nifti1Image(mask_np, affine=np.eye(4))
        safe_id = sample_id.replace("/", "_")
        nib.save(img, out_dir / f"{safe_id}_pred.nii.gz")


if __name__ == "__main__":
    main()
