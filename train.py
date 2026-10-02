import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP

from data.dataset import build_dataloader
from models.cross_attention import CrossAttentionFusion
from models.grounder import Grounder
from training.losses import CombinedLoss
from training.trainer import Trainer, build_optimizer, build_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D Visual Grounder")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--override",
        nargs="*",
        metavar="KEY=VALUE",
        help="Dot-notation config overrides, e.g. training.batch_size=2",
    )
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--resume-partial", action="store_true",
        help="Warm-start model weights from --resume with strict=False instead of resuming a "
             "run in progress -- for loading a checkpoint from before an architecture change "
             "(e.g. matching-shape UNet backbone weights carry over; new keys stay at their "
             "fresh init). Optimizer/scheduler/epoch/best_dice are NOT restored in this mode.",
    )
    parser.add_argument(
        "--init-text-proj", default=None,
        help="Path to a saved text_proj state_dict (Linear+LayerNorm+Dropout) to broadcast-load "
             "into every CrossAttentionFusion's own text_proj as a shared starting point, instead "
             "of each stage's random init. Applied before --resume, if both given, so a resumed "
             "checkpoint's own weights always take precedence.",
    )
    parser.add_argument(
        "--epochs-this-job", type=int, default=None,
        help="Cap how many epochs THIS process runs, without touching "
             "training.num_epochs (the fixed cosine-schedule horizon/ceiling) -- "
             "for SLURM chain resubmission, one epoch per job (see "
             "submission_scripts/finetuning/submit_h200_4gpu.sh). Omit for the "
             "traditional single-job behavior: run until early-stop or num_epochs.",
    )
    parser.add_argument(
        "--override-lr", type=float, default=None,
        help="Change the base LR when resuming a run in progress (--resume without "
             "--resume-partial), preserving epoch count/optimizer momentum/best_dice -- unlike "
             "editing configs/*.yaml's optimizer.lr, which a strict resume's "
             "optimizer.load_state_dict()/scheduler.load_state_dict() would silently overwrite "
             "back to the checkpoint's saved value anyway (and which also renames "
             "checkpoint.output_dir via _build_run_name, pointing auto-resume at an empty dir).",
    )
    return parser.parse_args()


def _broadcast_text_proj_init(model: torch.nn.Module, state_dict: dict) -> int:
    """Load the same text_proj weights into every CrossAttentionFusion in the model.

    Each stage starts identical, then diverges under independent gradients during
    training. Returns how many modules were initialized.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, CrossAttentionFusion):
            module.text_proj.load_state_dict(state_dict)
            n += 1
    return n


def _build_run_name(cfg: dict) -> str:
    ch = cfg["model"]["unet_base_channels"]
    bs = cfg["training"]["batch_size"]
    lr = float(cfg["optimizer"]["lr"])
    fusion = cfg["model"].get("fusion_type", "cross_attention")
    if cfg["model"].get("encoder_type", "unet") == "merlin":
        # spatial_mode/spatial_size are meaningless for the Merlin-grid path
        # (see models/merlin_encoder.py::MerlinEncoder.FULL_GRID_HWD) --
        # fixed token instead of the plain-UNet path's own grid description.
        grid = "merlin224x224x160"
    else:
        mode = cfg["data"]["spatial_mode"]
        d, h, w = cfg["data"]["spatial_size"]
        grid = f"{mode}{d}x{h}x{w}"
    name = f"ch{ch}_bs{bs}_lr{lr:.0e}_{grid}_{fusion}"
    if cfg["data"].get("group_by_image"):
        # Distinct run dir so --resume's auto-discovery can't load a
        # pre-group_by_image checkpoint into a run whose epoch/step
        # bookkeeping now means something different (see build_dataloader).
        name += "_grpimg"
    if cfg["data"].get("multi_window"):
        name += "_mw"
    max_samples = cfg["data"].get("max_samples")
    if max_samples is not None:
        name += f"_max{max_samples}"
    return name


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    # A mid-epoch checkpoint always represents more progress than any
    # completed-epoch one -- it's only ever written during an epoch that
    # hasn't finished yet, and gets deleted once that epoch does.
    mid_epoch = ckpt_dir / "mid_epoch.pt"
    if mid_epoch.exists():
        return mid_epoch
    checkpoints = sorted(
        ckpt_dir.glob("epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return checkpoints[-1] if checkpoints else None


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for kv in overrides:
        key, _, raw_val = kv.partition("=")
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        # Attempt YAML type coercion
        val = yaml.safe_load(raw_val)
        node[parts[-1]] = val
    return cfg


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.override:
        cfg = apply_overrides(cfg, args.override)

    seed = cfg.get("seed", 42)
    _seed_everything(seed)

    # DDP setup
    if _is_ddp():
        rank, local_rank, world_size = _setup_ddp()
        device = torch.device("cuda", local_rank)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = (rank == 0)
    if not is_main:
        logging.disable(logging.CRITICAL)

    logger.info(f"Using device: {device}  rank={rank}/{world_size}")

    encoder_type = cfg["model"].get("encoder_type", "unet")
    merlin_cfg = cfg["model"].get("merlin", {})
    if encoder_type == "merlin" and cfg["data"].get("multi_window"):
        logger.warning(
            "encoder_type=merlin forces in_channels=1 (Merlin always triplicates a single "
            "channel internally) -- data.multi_window=true is set but has no effect for this "
            "encoder path."
        )
    if encoder_type == "merlin":
        # embedding_cache lives under data.merlin (a data-pipeline concern), not
        # model.merlin, and falls back to the top-level key when unset.
        effective_embedding_cache = cfg["data"].get("merlin", {}).get("embedding_cache") or cfg["data"].get("embedding_cache")
    else:
        effective_embedding_cache = cfg["data"].get("embedding_cache")

    raw_model = Grounder(
        text_encoder_name=cfg["model"]["text_encoder_name"],
        text_proj_dim=cfg["model"]["text_proj_dim"],
        freeze_text_encoder=cfg["model"]["freeze_text_encoder"],
        finetune_last_n_layers=cfg["model"].get("finetune_last_n_layers", 0),
        unet_base_channels=cfg["model"]["unet_base_channels"],
        unet_channel_mult=cfg["model"]["unet_channel_mult"],
        num_heads=cfg["model"]["num_heads"],
        target_q_tokens=cfg["model"]["target_q_tokens"],
        dropout=cfg["model"]["dropout"],
        in_channels=1 if encoder_type == "merlin" else (3 if cfg["data"].get("multi_window") else 1),
        spatial_size=tuple(cfg["data"]["spatial_size"]),
        load_text_backbone=not effective_embedding_cache,
        fusion_type=cfg["model"].get("fusion_type", "cross_attention"),
        voxtell_guidance_dim=cfg["model"].get("voxtell", {}).get("guidance_dim", 32),
        voxtell_prompt_decoder_dim=cfg["model"].get("voxtell", {}).get("prompt_decoder_dim", 256),
        voxtell_prompt_decoder_layers=cfg["model"].get("voxtell", {}).get("prompt_decoder_layers", 6),
        voxtell_prompt_decoder_heads=cfg["model"].get("voxtell", {}).get("prompt_decoder_heads", 8),
        freeze_language_projectors=cfg["model"].get("freeze_language_projectors", True),
        encoder_type=encoder_type,
        merlin_model_dir=merlin_cfg.get("model_dir", ""),
        merlin_clinical_longformer_dir=merlin_cfg.get("clinical_longformer_dir", ""),
        freeze_merlin_encoder=merlin_cfg.get("freeze_merlin_encoder", True),
        finetune_last_n_merlin_stages=merlin_cfg.get("finetune_last_n_merlin_stages", 0),
    ).to(device)

    if is_main:
        n_params = sum(p.numel() for p in raw_model.parameters()) / 1e6
        n_trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad) / 1e6
        logger.info(f"Model: {n_params:.1f}M total params, {n_trainable:.1f}M trainable")

    if args.init_text_proj:
        state_dict = torch.load(args.init_text_proj, map_location=device)
        n = _broadcast_text_proj_init(raw_model, state_dict)
        if is_main:
            logger.info(f"Broadcast-loaded text_proj weights from {args.init_text_proj} into "
                        f"{n} cross-attention module(s)")

    raw_model.unet.grad_checkpointing = cfg["training"].get("gradient_checkpointing", False)

    if cfg["training"].get("compile", False):
        if is_main:
            logger.info("Compiling model with torch.compile...")
        raw_model = torch.compile(raw_model)

    if effective_embedding_cache and is_main:
        logger.info("embedding_cache set — text encoder backbone not loaded")

    if _is_ddp():
        raw_model = SyncBatchNorm.convert_sync_batchnorm(raw_model)
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = raw_model

    if encoder_type == "merlin":
        from data.merlin_dataset import build_merlin_dataloader
        _build_loader = build_merlin_dataloader
    else:
        _build_loader = build_dataloader

    train_loader = _build_loader(cfg["data"]["train_manifest"], cfg, split="train",
                                 rank=rank, world_size=world_size, seed=seed)
    val_loader = _build_loader(cfg["data"]["val_manifest"], cfg, split="val",
                               rank=rank, world_size=world_size, seed=seed)

    # Optional: Macro Hit Rate reporting (see Trainer.macro_hit_rate_epoch) needs
    # ED and ONC validation samples kept separate from val_loader's merged set --
    # absent (None) for configs with no ED/ONC data (e.g. rexgroundingct_finetune.yaml).
    ed_val_manifest = cfg["data"].get("ed_val_manifest")
    onc_val_manifest = cfg["data"].get("onc_val_manifest")
    ed_val_loader = (
        _build_loader(ed_val_manifest, cfg, split="val", rank=rank, world_size=world_size, seed=seed)
        if ed_val_manifest else None
    )
    onc_val_loader = (
        _build_loader(onc_val_manifest, cfg, split="val", rank=rank, world_size=world_size, seed=seed)
        if onc_val_manifest else None
    )

    # Trainer overwrites dice_weight/bce_weight every epoch per the warmup ramp
    # (see Trainer._update_loss_weights); start at dice_weight_start here just
    # for a sane initial value before the first epoch sets it.
    dice_weight_start = float(cfg["loss"]["dice_weight_start"])
    loss_fn = CombinedLoss(
        dice_weight=dice_weight_start,
        bce_weight=1.0 - dice_weight_start,
        bce_pos_weight=cfg["loss"].get("bce_pos_weight"),
    ).to(device)

    optimizer = build_optimizer(raw_model, cfg)
    scheduler = build_scheduler(optimizer, cfg, num_training_steps=len(train_loader) * cfg["training"]["num_epochs"])

    run_name = _build_run_name(cfg)
    if is_main:
        logger.info(f"Run name: {run_name}")

    trainer = Trainer(
        model=model,
        raw_model=raw_model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        output_dir=Path(cfg["checkpoint"]["output_dir"]) / run_name,
        use_amp=cfg["training"].get("use_amp", True),
        rank=rank,
        world_size=world_size,
        ed_val_loader=ed_val_loader,
        onc_val_loader=onc_val_loader,
    )

    resume_path = args.resume
    if resume_path is None:
        latest = _find_latest_checkpoint(trainer.output_dir / "checkpoints")
        if latest is not None:
            resume_path = str(latest)
            if is_main:
                logger.info(f"Auto-resuming from latest checkpoint: {resume_path}")
        elif is_main:
            logger.warning(
                f"No checkpoint found under {trainer.output_dir / 'checkpoints'} -- "
                f"starting from epoch 0. If you expected to resume a run, check whether "
                f"run_name (derived from model/training/data config) changed since the "
                f"checkpoint you expected was written."
            )

    if resume_path:
        trainer.load_checkpoint(resume_path, strict=not args.resume_partial)
    if args.override_lr is not None:
        trainer.override_lr(args.override_lr)
    if _is_ddp():
        dist.barrier()

    try:
        finished = trainer.fit(cfg["training"]["num_epochs"], max_epochs_this_call=args.epochs_this_job)
        if is_main and finished:
            (trainer.output_dir / "checkpoints" / "TRAINING_COMPLETE").touch()
    finally:
        _teardown_ddp()


def _is_ddp() -> bool:
    return "LOCAL_RANK" in os.environ


def _setup_ddp() -> tuple[int, int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size()


def _teardown_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
