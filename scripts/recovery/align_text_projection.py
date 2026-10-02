#!/usr/bin/env python3
"""
Re-aligns the text_encoder.projection layer to an already-trained checkpoint's
UNet, after the original projection used to build embeddings_mmap was lost
(see precompute_embeddings.py's docstring -- checkpoints trained with
embedding_cache set never persist the projection, so it only ever exists
transiently in whatever process last ran precompute_embeddings.py).

Loads the checkpoint's UNet weights, freezes them, and trains ONLY the
(freshly-initialized) projection layer with live text encoding against the
normal train manifest and Dice+BCE loss, so the projection learns to produce
features the frozen UNet already knows how to interpret. The Qwen backbone
stays frozen throughout (same as normal training).

Once this converges, use precompute_embeddings.py --load-projection on its
output to regenerate every embedding cache (main + any held-out test sets)
consistently, then resume normal embedding_cache-mode training.

Usage
-----
    python align_text_projection.py --config configs/default.yaml \
        --checkpoint runs/h200/checkpoints/best.pt \
        --output text_projection_aligned.pt --steps 2000
"""
import argparse
import logging
import sys

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import build_dataloader
from models.grounder import Grounder
from train import apply_overrides
from training.losses import CombinedLoss
from training.trainer import _text_kwargs, _trim_padding
from utils.metrics import dice_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Already-trained checkpoint to align against")
    parser.add_argument("--output", default="text_projection_aligned.pt")
    parser.add_argument("--steps", type=int, default=2000,
                         help="Optimizer updates, not raw samples (see --grad-accum-steps).")
    parser.add_argument("--grad-accum-steps", type=int, default=4,
                         help="training.batch_size is 1, so a single sample's loss/dice is very "
                              "noisy -- accumulate gradients over this many samples per optimizer "
                              "update to smooth it out.")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3,
                         help="Projection is training from a random init, not fine-tuning an "
                              "already-good representation -- optimizer.text_lr_scale (meant for "
                              "the latter) is far too small for this. Default 1e-3. Decayed to "
                              "1%% of this via cosine annealing over --steps.")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Dot-notation config overrides, e.g. model.text_encoder_name=/path/to/model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Grounder(
        text_encoder_name=cfg["model"]["text_encoder_name"],
        text_feat_dim=cfg["model"]["text_feat_dim"],
        freeze_text_encoder=True,
        finetune_last_n_layers=0,  # only the projection trains here, not extra Qwen layers
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
        load_text_backbone=True,  # need live encoding so gradients reach the projection
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"Checkpoint has unexpected keys this model doesn't have: {unexpected}")
    non_text_missing = [k for k in missing if not k.startswith("text_encoder.")]
    if not missing or non_text_missing:
        raise RuntimeError(
            f"Expected only text_encoder.* keys missing (checkpoint was trained with "
            f"embedding_cache set, so it never had a projection layer). "
            f"Unexpected non-text_encoder missing key(s): {non_text_missing}"
        )
    logger.info(f"Loaded UNet weights from {args.checkpoint} (epoch {ckpt.get('epoch', '?')}); "
                f"{len(missing)} fresh text_encoder.* key(s) left at their pretrained/random init, as expected "
                f"(Qwen backbone keeps its pretrained weights, projection stays randomly initialized to be trained)")

    for p in model.unet.parameters():
        p.requires_grad_(False)
    model.unet.grad_checkpointing = cfg["training"].get("gradient_checkpointing", False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info(f"Training {n_trainable} projection param(s), UNet and Qwen backbone frozen")

    cfg_live = dict(cfg)
    cfg_live["data"] = dict(cfg["data"])
    cfg_live["data"]["embedding_cache"] = ""  # force live tokenization -- need gradients into the projection
    train_loader = build_dataloader(cfg["data"]["train_manifest"], cfg_live, split="train",
                                     num_workers=cfg["data"]["num_workers"])

    # No epoch/warmup concept in this short alignment run -- use the final
    # target weighting the main training run converges to (see Trainer._update_loss_weights).
    dice_weight_end = float(cfg["loss"]["dice_weight_end"])
    loss_fn = CombinedLoss(
        dice_weight=dice_weight_end,
        bce_weight=1.0 - dice_weight_end,
        bce_pos_weight=cfg["loss"].get("bce_pos_weight"),
    ).to(device)

    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=float(opt_cfg.get("weight_decay", 1e-5)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.01)
    grad_clip = cfg["training"].get("grad_clip_norm", 1.0)
    use_amp = cfg["training"].get("use_amp", True) and device.type == "cuda"

    model.train()
    total_loss = total_dice = 0.0
    n = 0
    step = 0
    micro_step = 0
    pbar = tqdm(total=args.steps, desc="Aligning projection")
    optimizer.zero_grad()
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(image, **_text_kwargs(batch, device))
                logits, mask = _trim_padding(logits, mask, batch["pad_amounts"])
                loss, loss_parts = loss_fn(logits, mask)

            (loss / args.grad_accum_steps).backward()
            micro_step += 1

            bs = image.size(0)
            d = dice_score(logits.detach(), mask, from_logits=True)
            total_loss += loss.item() * bs
            total_dice += d.sum().item()
            n += bs

            if micro_step % args.grad_accum_steps != 0:
                continue

            nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1
            pbar.update(1)

            if step % args.log_every == 0:
                logger.info(f"step {step}/{args.steps}  lr={scheduler.get_last_lr()[0]:.2e}  loss={loss.item():.4f}  "
                            f"avg_loss={total_loss / n:.4f}  avg_dice={total_dice / n:.4f}  "
                            f"dice_loss={loss_parts['dice']:.4f}")
    pbar.close()

    torch.save(model.text_encoder.projection.state_dict(), args.output)
    logger.info(f"Final {args.steps} steps: avg_loss={total_loss / n:.4f}  avg_dice={total_dice / n:.4f}")
    logger.info(f"Saved aligned projection → {args.output}")


if __name__ == "__main__":
    main()
