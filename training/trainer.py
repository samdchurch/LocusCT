import json
import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import dice_score, iou_score

logger = logging.getLogger(__name__)

# ED validation's raw manifest "finding" strings, grouped into 13 clinical categories
# (confirmed with the authors: Calcification/Calculus/Calculi/Stone are one group, Cyst/Cystic
# Lesion are one group, Lymphadenopathy/Lymph Node are one group; Gallstone stays its
# own category despite also being a "calculus"). Used only for ed_macro_hit_rate below
# -- every value in curated_ed_val_data.json's "finding" field must appear here as a
# key, or macro_hit_rate_epoch's dict lookup raises KeyError naming the missing one.
ED_CATEGORIES = [
    "Abscess", "Aneurysm", "Appendicitis", "Calculus/Stone", "Cyst", "Diverticulitis",
    "Gallstone", "Hematoma", "Hernia", "Lesion", "Lymphadenopathy", "Mass", "Nodule",
]
FINDING_TO_ED_CATEGORY = {
    "Abscess": "Abscess",
    "Aneurysm": "Aneurysm",
    "Appendicitis": "Appendicitis",
    "Calcification": "Calculus/Stone",
    "Calculus": "Calculus/Stone",
    "Calculi": "Calculus/Stone",
    "Stone": "Calculus/Stone",
    "Cyst": "Cyst",
    "Cystic Lesion": "Cyst",
    "Diverticulitis": "Diverticulitis",
    "Gallstone": "Gallstone",
    "Hematoma": "Hematoma",
    "Hernia": "Hernia",
    "Lesion": "Lesion",
    "Lymphadenopathy": "Lymphadenopathy",
    "Lymph Node": "Lymphadenopathy",
    "Mass": "Mass",
    "Nodule": "Nodule",
}


def _rng_state() -> dict:
    state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _reduce_metrics(
    total_loss: float,
    total_dice: float,
    total_iou: float,
    total_hits: float,
    n: int,
    device: torch.device,
    world_size: int,
) -> dict[str, float]:
    if world_size > 1:
        t = torch.tensor(
            [total_loss, total_dice, total_iou, total_hits, float(n)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, total_dice, total_iou, total_hits, n = t.tolist()
    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
        "hit_rate": total_hits / n,
    }


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        raw_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
        output_dir: Path,
        use_amp: bool = True,
        rank: int = 0,
        world_size: int = 1,
        ed_val_loader: DataLoader | None = None,
        onc_val_loader: DataLoader | None = None,
    ) -> None:
        self.model = model
        self.raw_model = raw_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Optional -- absent (both None) for configs with no ED/ONC data (e.g.
        # rexgroundingct_finetune.yaml), in which case macro_hit_rate_epoch is a no-op.
        self.ed_val_loader = ed_val_loader
        self.onc_val_loader = onc_val_loader
        self._ed_category_by_mask: dict[str, str] = {}
        if ed_val_loader is not None:
            for sample in ed_val_loader.dataset.samples:
                self._ed_category_by_mask[sample["mask"]] = FINDING_TO_ED_CATEGORY[sample["finding"]]
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.use_amp = use_amp
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        self.grad_clip = cfg["training"].get("grad_clip_norm", 1.0)
        self.log_every = cfg["logging"].get("log_every_n_steps", 50)
        self.checkpoint_every_n_steps = cfg["training"].get("checkpoint_every_n_steps")
        self.dice_weight_start = float(cfg["loss"]["dice_weight_start"])
        self.dice_weight_end = float(cfg["loss"]["dice_weight_end"])
        self.loss_warmup_epochs = cfg["scheduler"].get("warmup_epochs", 0)
        self.best_dice = 0.0
        self.start_epoch = 0
        self.resume_step_in_epoch = 0
        # Early stopping on Macro Hit Rate (see macro_hit_rate_epoch) -- null/0 disables
        # it, training then always runs the full num_epochs (also the case whenever
        # ed_val_loader/onc_val_loader aren't set, since macro_hit_rate_epoch returns {}
        # and fit() only checks this against a non-empty result).
        self.early_stop_patience = cfg["training"].get("early_stop_patience")
        self.best_macro_hit_rate = 0.0
        self.epochs_without_macro_improvement = 0

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        self.metrics_log_path = output_dir / "metrics.jsonl"
        # Per-rank file under DDP -- unlike metrics.jsonl (all-reduced, rank-0-only),
        # each rank sees a different shard of samples, so gating this on is_main would
        # silently drop every sample processed by non-zero ranks.
        self.sample_loss_log_path = (
            output_dir / "sample_losses.jsonl" if world_size == 1
            else output_dir / f"sample_losses_rank{rank}.jsonl"
        )

        self._wandb = None
        if self.is_main and cfg["logging"].get("use_wandb", False):
            import wandb
            wandb.init(project=cfg["logging"]["project"], config=cfg)
            self._wandb = wandb

    def _log_json(self, record: dict) -> None:
        if not self.is_main:
            return
        record = {"time": datetime.now(timezone.utc).isoformat(), **record}
        with open(self.metrics_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _log_sample_losses(
        self,
        phase: str,
        epoch: int,
        step: int,
        ids: list[str],
        per_sample_loss: torch.Tensor,
        per_sample_dice: torch.Tensor,
    ) -> None:
        """One line per sample, so per-sample loss can be tracked over time
        to spot samples that aren't improving (see sample_losses.jsonl)."""
        time_str = datetime.now(timezone.utc).isoformat()
        with open(self.sample_loss_log_path, "a") as f:
            for sample_id, loss_val, dice_val in zip(ids, per_sample_loss.tolist(), per_sample_dice.tolist()):
                f.write(json.dumps({
                    "time": time_str,
                    "phase": phase,
                    "epoch": epoch,
                    "step": step,
                    "sample_id": sample_id,
                    "loss": loss_val,
                    "dice_score": dice_val,
                }) + "\n")

    def _update_loss_weights(self, epoch: int) -> None:
        """
        Ramp dice_weight linearly from dice_weight_start to dice_weight_end over
        loss_warmup_epochs (mirroring the LR scheduler's warmup_epochs), then hold
        at dice_weight_end. bce_weight is always its complement. Stateless in
        epoch alone, so this needs no checkpoint save/restore on resume.
        """
        if self.loss_warmup_epochs > 0:
            frac = min(1.0, epoch / self.loss_warmup_epochs)
        else:
            frac = 1.0
        dice_weight = self.dice_weight_start + frac * (self.dice_weight_end - self.dice_weight_start)
        self.loss_fn.dice_weight = dice_weight
        self.loss_fn.bce_weight = 1.0 - dice_weight

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self._update_loss_weights(epoch)
        self.model.train()
        total_loss = total_dice = total_iou = total_hits = 0.0
        n = 0

        # Only nonzero right after loading a mid-epoch checkpoint, and only for
        # the resumed epoch -- consumed here so later epochs start at step 0.
        resume_step = self.resume_step_in_epoch
        self.resume_step_in_epoch = 0

        # .batch_size is None under group_by_image (DataLoader batch_size=None
        # mode) -- each step already consumes exactly one sampler index there,
        # so `or 1` makes the skip_samples math below a no-op in that case
        # instead of `resume_step * None` raising.
        batch_size = self.train_loader.batch_size or 1
        sampler = self.train_loader.sampler
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch, skip_samples=resume_step * batch_size)

        step = resume_step
        total_steps = resume_step + len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False, disable=not self.is_main,
                    initial=resume_step, total=total_steps)
        for batch in pbar:
            image = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                logits = self.model(image, **_text_kwargs(batch, self.device))
                # Trim padding if present (fixed spatial_mode)
                logits, mask = _trim_padding(logits, mask, batch["pad_amounts"])
                loss, loss_parts = self.loss_fn(logits, mask)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # mask.size(0), not image.size(0): under group_by_image, image is
            # batch 1 (one volume) while mask/logits are batch N (findings for
            # that image) -- they're only ever equal in the old per-triplet mode.
            bs = mask.size(0)
            d = dice_score(logits.detach(), mask, from_logits=True)
            total_loss += loss.item() * bs
            total_dice += d.sum().item()
            total_iou += iou_score(logits.detach(), mask, from_logits=True).sum().item()
            total_hits += (d >= 0.1).sum().item()
            n += bs
            step += 1

            self._log_sample_losses("train", epoch, step, batch["id"], loss_parts["per_sample"], d)

            if step % self.log_every == 0:
                avg_loss = total_loss / n
                avg_dice_score = total_dice / n
                logger.info(
                    f"Epoch {epoch} step {step}/{total_steps} "
                    f"loss={loss.item():.4f} avg_loss={avg_loss:.4f} "
                    f"dice_loss={loss_parts['dice']:.4f} avg_dice_score={avg_dice_score:.4f}"
                )
                if self._wandb:
                    self._wandb.log({"train/loss": loss.item(), "train/avg_loss": avg_loss,
                                      **{f"train/{k}": v for k, v in loss_parts.items() if k != "per_sample"}})
                self._log_json({
                    "phase": "train_step",
                    "epoch": epoch,
                    "step": step,
                    "loss": loss.item(),
                    "avg_loss": avg_loss,
                    "dice_loss": loss_parts["dice"],
                    "avg_dice_score": avg_dice_score,
                })

            if (self.is_main and self.checkpoint_every_n_steps
                    and step % self.checkpoint_every_n_steps == 0):
                self.save_mid_epoch_checkpoint(epoch, step)

        return _reduce_metrics(total_loss, total_dice, total_iou, total_hits, n, self.device, self.world_size)

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        total_loss = total_dice = total_iou = total_hits = 0.0
        n = 0
        step = 0

        for batch in tqdm(self.val_loader, desc=f"Val {epoch}", leave=False, disable=not self.is_main):
            image = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                logits = self.model(image, **_text_kwargs(batch, self.device))
                logits, mask = _trim_padding(logits, mask, batch["pad_amounts"])
                loss, loss_parts = self.loss_fn(logits, mask)

            bs = image.size(0)
            d = dice_score(logits, mask, from_logits=True)
            total_loss += loss.item() * bs
            total_dice += d.sum().item()
            total_iou += iou_score(logits, mask, from_logits=True).sum().item()
            total_hits += (d >= 0.1).sum().item()
            n += bs
            step += 1

            self._log_sample_losses("val", epoch, step, batch["id"], loss_parts["per_sample"], d)

        return _reduce_metrics(total_loss, total_dice, total_iou, total_hits, n, self.device, self.world_size)

    @torch.no_grad()
    def _dice_per_sample(self, loader: DataLoader, desc: str) -> list[tuple[str, float]]:
        """(sample_id, dice) for every sample in `loader` that THIS rank sees --
        callers needing a global result must all_reduce, same as val_epoch/
        _reduce_metrics does for its own scalar totals."""
        self.model.eval()
        results: list[tuple[str, float]] = []
        for batch in tqdm(loader, desc=desc, leave=False, disable=not self.is_main):
            image = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                logits = self.model(image, **_text_kwargs(batch, self.device))
                logits, mask = _trim_padding(logits, mask, batch["pad_amounts"])
            d = dice_score(logits, mask, from_logits=True)
            results.extend(zip(batch["id"], d.tolist()))
        return results

    @torch.no_grad()
    def macro_hit_rate_epoch(self, epoch: int) -> dict[str, float]:
        """Macro Hit Rate: ONC's is the plain hit rate (dice >= 0.1) across all ONC
        val samples; ED's is the MEAN of the per-category hit rate across
        ED_CATEGORIES (13 categories weighted equally, not sample-weighted) --
        see FINDING_TO_ED_CATEGORY. The reported macro_hit_rate is the average of
        those two. No-op ({}) if ed_val_loader/onc_val_loader weren't given (e.g.
        rexgroundingct_finetune.yaml, which has no ED/ONC data)."""
        if self.ed_val_loader is None or self.onc_val_loader is None:
            return {}

        onc_results = self._dice_per_sample(self.onc_val_loader, desc=f"ONC hit {epoch}")
        onc_hits = float(sum(1 for _, d in onc_results if d >= 0.1))
        onc_n = float(len(onc_results))

        ed_results = self._dice_per_sample(self.ed_val_loader, desc=f"ED hit {epoch}")
        cat_hits = torch.zeros(len(ED_CATEGORIES), dtype=torch.float64, device=self.device)
        cat_counts = torch.zeros(len(ED_CATEGORIES), dtype=torch.float64, device=self.device)
        for sample_id, d in ed_results:
            idx = ED_CATEGORIES.index(self._ed_category_by_mask[sample_id])
            cat_counts[idx] += 1
            if d >= 0.1:
                cat_hits[idx] += 1

        if self.world_size > 1:
            onc_t = torch.tensor([onc_hits, onc_n], dtype=torch.float64, device=self.device)
            dist.all_reduce(onc_t, op=dist.ReduceOp.SUM)
            onc_hits, onc_n = onc_t.tolist()
            dist.all_reduce(cat_hits, op=dist.ReduceOp.SUM)
            dist.all_reduce(cat_counts, op=dist.ReduceOp.SUM)

        if (cat_counts == 0).any():
            missing = [c for c, n in zip(ED_CATEGORIES, cat_counts.tolist()) if n == 0]
            raise RuntimeError(
                f"macro_hit_rate_epoch: no ED validation samples for categor{'y' if len(missing) == 1 else 'ies'} "
                f"{missing} -- check data.ed_val_manifest"
            )

        onc_macro_hit_rate = onc_hits / onc_n
        ed_macro_hit_rate = (cat_hits / cat_counts).mean().item()
        return {
            "onc_macro_hit_rate": onc_macro_hit_rate,
            "ed_macro_hit_rate": ed_macro_hit_rate,
            "macro_hit_rate": (onc_macro_hit_rate + ed_macro_hit_rate) / 2.0,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(self, num_epochs: int, max_epochs_this_call: int | None = None) -> bool:
        """Returns True once training has reached a terminal state (early-stop
        fired, or num_epochs reached); False if this call merely used up its
        max_epochs_this_call budget with epochs still remaining."""
        epochs_run = 0
        for epoch in range(self.start_epoch, num_epochs):
            # train_epoch() calls sampler.set_epoch(epoch, skip_samples=...) itself,
            # since it needs to pass the resumed step's skip_samples through too.

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            lr = self.optimizer.param_groups[-1]["lr"]
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.val_epoch(epoch)
            macro_metrics = self.macro_hit_rate_epoch(epoch)
            self.scheduler.step()

            # Early-stopping bookkeeping: macro_metrics is already identical on every
            # rank (all-reduced inside macro_hit_rate_epoch), so every rank computes
            # the same should_stop decision independently here -- no extra broadcast
            # needed, and every rank breaks the loop together below. Unlike this,
            # best_dice/checkpoint writing further down stays rank-0-only, since it
            # doesn't affect loop control flow.
            should_stop = False
            if macro_metrics:
                if macro_metrics["macro_hit_rate"] > self.best_macro_hit_rate:
                    self.best_macro_hit_rate = macro_metrics["macro_hit_rate"]
                    self.epochs_without_macro_improvement = 0
                else:
                    self.epochs_without_macro_improvement += 1
                if self.early_stop_patience and self.epochs_without_macro_improvement >= self.early_stop_patience:
                    should_stop = True

            if self.is_main:
                log_line = (
                    f"Epoch {epoch} | lr={lr:.2e} dice_weight={self.loss_fn.dice_weight:.3f} | "
                    f"train loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} hit={train_metrics['hit_rate']:.3f} | "
                    f"val   loss={val_metrics['loss']:.4f}  dice={val_metrics['dice']:.4f}  hit={val_metrics['hit_rate']:.3f}"
                )
                if macro_metrics:
                    log_line += (
                        f" | macro_hit={macro_metrics['macro_hit_rate']:.3f} "
                        f"(ed={macro_metrics['ed_macro_hit_rate']:.3f} onc={macro_metrics['onc_macro_hit_rate']:.3f})"
                    )
                logger.info(log_line)
                if should_stop:
                    logger.info(
                        f"Early stopping: macro_hit_rate hasn't improved for "
                        f"{self.epochs_without_macro_improvement} epoch(s) "
                        f"(patience={self.early_stop_patience}, best={self.best_macro_hit_rate:.3f})"
                    )
                peak_alloc = peak_reserved = None
                if self.device.type == "cuda":
                    peak_alloc = torch.cuda.max_memory_allocated(self.device) / 2**30
                    peak_reserved = torch.cuda.max_memory_reserved(self.device) / 2**30
                    logger.info(f"Epoch {epoch} | rank0 peak GPU memory: {peak_alloc:.2f} GiB allocated, {peak_reserved:.2f} GiB reserved")
                if self._wandb:
                    self._wandb.log(
                        {f"train/{k}": v for k, v in train_metrics.items()} |
                        {f"val/{k}": v for k, v in val_metrics.items()} |
                        {f"val/{k}": v for k, v in macro_metrics.items()} |
                        {"epoch": epoch, "lr": lr, "dice_weight": self.loss_fn.dice_weight}
                    )
                self._log_json({
                    "phase": "epoch",
                    "epoch": epoch,
                    "lr": lr,
                    "dice_weight": self.loss_fn.dice_weight,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"val_{k}": v for k, v in macro_metrics.items()},
                    "peak_gpu_alloc_gib": peak_alloc,
                    "peak_gpu_reserved_gib": peak_reserved,
                })

                is_best = val_metrics["dice"] > self.best_dice
                if is_best:
                    self.best_dice = val_metrics["dice"]

                self.save_checkpoint(epoch, val_metrics, is_best)

            if should_stop:
                break

            epochs_run += 1
            if max_epochs_this_call is not None and epochs_run >= max_epochs_this_call:
                return False

        return True

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        ckpt_dir = self.output_dir / "checkpoints"
        path = ckpt_dir / f"epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "step_in_epoch": 0,  # a completed-epoch checkpoint; see save_mid_epoch_checkpoint
                "model_state_dict": self.raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics": metrics,
                "best_dice": self.best_dice,
                "best_macro_hit_rate": self.best_macro_hit_rate,
                "epochs_without_macro_improvement": self.epochs_without_macro_improvement,
            },
            path,
        )
        if is_best:
            best_path = ckpt_dir / "best.pt"
            shutil.copy2(path, best_path)
            logger.info(f"New best checkpoint (dice={metrics['dice']:.4f}) → {best_path}")

        # Any mid-epoch checkpoint from this epoch is now superseded -- the
        # epoch it was a partial snapshot of has since finished normally.
        (ckpt_dir / "mid_epoch.pt").unlink(missing_ok=True)

        self._prune_checkpoints(ckpt_dir)

    def save_mid_epoch_checkpoint(self, epoch: int, step_in_epoch: int) -> None:
        """
        Rolling, single-file safety-net checkpoint written partway through an
        epoch (see Trainer.checkpoint_every_n_steps), so a killed/preempted
        job resumes without re-consuming already-seen samples. Overwritten
        each time -- unlike epoch_*.pt, there's no history to keep, just the
        latest point to resume from. Superseded (deleted) once the epoch it
        belongs to completes normally, via save_checkpoint.
        """
        ckpt_dir = self.output_dir / "checkpoints"
        path = ckpt_dir / "mid_epoch.pt"
        torch.save(
            {
                "epoch": epoch,
                "step_in_epoch": step_in_epoch,
                "model_state_dict": self.raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_dice": self.best_dice,
                "best_macro_hit_rate": self.best_macro_hit_rate,
                "epochs_without_macro_improvement": self.epochs_without_macro_improvement,
                "rng_state": _rng_state(),
            },
            path,
        )
        logger.info(f"Mid-epoch checkpoint: epoch {epoch} step {step_in_epoch} → {path}")

    def _prune_checkpoints(self, ckpt_dir: Path) -> None:
        keep = self.cfg["checkpoint"].get("keep_last_n", 3)
        if not keep:  # null/0 -- keep every epoch checkpoint, no pruning
            return
        checkpoints = sorted(
            [p for p in ckpt_dir.glob("epoch_*.pt")],
            key=lambda p: int(p.stem.split("_")[1]),
        )
        for old in checkpoints[:-keep]:
            old.unlink(missing_ok=True)

    def load_checkpoint(self, path: str, strict: bool = True) -> int:
        """
        strict=False warm-starts model weights only (e.g. resuming into an
        architecture change where matching-shape keys, like a UNet backbone
        trained before new text_proj submodules were added, should carry
        over). Optimizer/scheduler/epoch/best_dice are NOT restored in that
        case -- the old optimizer state's per-parameter buffers are keyed
        positionally, so loading them against a changed parameter set/order
        would silently misalign momentum to the wrong parameters. Training
        restarts bookkeeping from scratch, just with better initial weights
        than random.
        """
        ckpt = torch.load(path, map_location=self.device)
        if strict:
            self.raw_model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            # scaler_state_dict no longer used (bf16 needs no loss scaling)
            step_in_epoch = ckpt.get("step_in_epoch", 0)
            if step_in_epoch > 0:
                # Mid-epoch checkpoint: resume the SAME epoch (it isn't finished),
                # not epoch + 1, and pick up train_epoch() partway through via
                # resume_step_in_epoch. RNG state restore is best-effort: it
                # covers the main process (dropout, and augmentation when
                # num_workers=0), not persistent DataLoader worker processes,
                # which reseed at process start and can't be restored exactly.
                self.start_epoch = ckpt["epoch"]
                self.resume_step_in_epoch = step_in_epoch
                if "rng_state" in ckpt:
                    _restore_rng_state(ckpt["rng_state"])
                logger.info(f"Resumed from {path} (epoch {ckpt['epoch']}, mid-epoch step {step_in_epoch})")
            else:
                self.start_epoch = ckpt["epoch"] + 1
                self.resume_step_in_epoch = 0
                logger.info(f"Resumed from {path} (epoch {ckpt['epoch']})")
            self.best_dice = ckpt.get("best_dice", self.best_dice)
            self.best_macro_hit_rate = ckpt.get("best_macro_hit_rate", self.best_macro_hit_rate)
            self.epochs_without_macro_improvement = ckpt.get(
                "epochs_without_macro_improvement", self.epochs_without_macro_improvement
            )
        else:
            missing, unexpected = self.raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            logger.info(f"Partially warm-started weights from {path} (epoch {ckpt['epoch']}): "
                        f"{len(missing)} missing key(s), {len(unexpected)} unexpected key(s) -- "
                        f"optimizer/scheduler/epoch NOT restored, starting those fresh")
            if missing:
                logger.info(f"  missing: {missing}")
            if unexpected:
                logger.info(f"  unexpected: {unexpected}")
        return ckpt["epoch"]

    def override_lr(self, new_base_lr: float) -> None:
        """
        Rescale every optimizer param group's LR to new_base_lr, preserving
        each group's ratio to the base (unet) group -- e.g. text_encoder's
        reduced text_lr_scale stays proportionally reduced, just off the new
        base instead of the old one. Also patches the scheduler's base_lrs,
        since schedulers recompute and overwrite the optimizer's LR from
        their own base_lrs on every .step() call -- patching param_groups
        alone would get silently stomped on at the next epoch boundary.

        Epoch count, optimizer momentum (Adam's running averages), and
        best_dice are untouched -- this only changes the LR trajectory going
        forward, unlike --resume-partial which resets all of that to restart
        bookkeeping from scratch.
        """
        old_base = self.optimizer.param_groups[-1]["lr"]  # unet group is always last, see build_optimizer
        ratio = new_base_lr / old_base
        for i, g in enumerate(self.optimizer.param_groups):
            g["lr"] *= ratio
            if hasattr(self.scheduler, "base_lrs"):
                self.scheduler.base_lrs[i] *= ratio
        logger.info(f"Overrode LR: base {old_base:.2e} -> {new_base_lr:.2e} "
                    f"(all {len(self.optimizer.param_groups)} param group(s) scaled {ratio:.4g}x)")


# ------------------------------------------------------------------
# Optimizer / scheduler factories
# ------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """
    Up to three param groups: text encoder at reduced LR, Merlin encoder (if
    present) at its own reduced LR, UNet (base group) at full LR. Frozen
    parameters are excluded automatically (requires_grad=False) -- except
    each stage's text_proj (model.freeze_language_projectors), kept in
    unet_params even when frozen. Excluding it would shrink that param
    group, breaking optimizer.load_state_dict() on any checkpoint saved
    before it was frozen ("parameter group doesn't match the size of
    optimizer's group") -- unlike excluding the (potentially 8B-param)
    frozen text encoder, which matters for memory, text_proj is a few
    million params at most, so there's no real cost to just leaving it in
    the group; AdamW.step() already skips params whose .grad is None.

    IMPORTANT: unet_params (the base group) must stay LAST -- Trainer.
    override_lr and Trainer.fit both hardcode param_groups[-1] assuming the
    base group is last.
    """
    opt_cfg = cfg["optimizer"]
    # float() rather than trusting the YAML/--override type coercion directly --
    # PyYAML's implicit float resolver doesn't recognize every valid float
    # literal (e.g. exponents without a decimal point), silently leaving such
    # values as strings and turning this multiply into a confusing
    # "can't multiply sequence" error instead of a clear one.
    base_lr = float(opt_cfg["lr"])
    text_lr = base_lr * float(opt_cfg.get("text_lr_scale", 0.01))

    text_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
    # text_params is only ever non-empty when finetune_last_n_layers > 0 unfreezes some
    # of the frozen Qwen backbone -- the per-stage text projections (k_proj/v_proj/q_proj
    # in each cross-attention module) live in model.unet, not model.text_encoder, so they
    # already train at the full base_lr via unet_params, not the reduced text_lr.

    # Merlin encoder (model.encoder_type=="merlin"): only nonempty when
    # finetune_last_n_merlin_stages > 0 unfreezes some ResNet stages. The
    # proj_e0..proj_e4 channel-projection adapters are deliberately EXCLUDED
    # here (name check below) -- they're new decoder-adjacent layers, not
    # "Merlin finetuning", so they train at the normal base_lr via
    # unet_params instead, like every other newly-initialized UNet layer.
    merlin_lr = base_lr * float(opt_cfg.get("merlin_lr_scale", 0.01))

    def _is_merlin_backbone_param(name: str) -> bool:
        return name.startswith("merlin_encoder.") and "proj_" not in name

    merlin_params = [
        p for name, p in model.unet.named_parameters()
        if p.requires_grad and _is_merlin_backbone_param(name)
    ]
    unet_params = [
        p for name, p in model.unet.named_parameters()
        if (p.requires_grad or "text_proj" in name) and not _is_merlin_backbone_param(name)
    ]

    param_groups = []
    if text_params:
        param_groups.append({"params": text_params, "lr": text_lr})
    if merlin_params:
        param_groups.append({"params": merlin_params, "lr": merlin_lr})
    param_groups.append({"params": unet_params, "lr": base_lr})  # MUST stay last -- see override_lr

    return torch.optim.AdamW(
        param_groups,
        weight_decay=float(opt_cfg.get("weight_decay", 1e-5)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    num_training_steps: int,
) -> Any:
    sched_cfg = cfg["scheduler"]
    name = sched_cfg.get("name", "cosine")
    num_epochs = cfg["training"]["num_epochs"]

    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    if name == "linear_warmup_cosine":
        warmup_epochs = sched_cfg.get("warmup_epochs", 5)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
            import math
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unknown scheduler: {name}")


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _text_kwargs(batch: dict, device: torch.device) -> dict:
    """Return the right keyword args for Grounder.forward() based on what the batch contains."""
    if "text_feats" in batch:
        return {
            "text_feats": batch["text_feats"].to(device, non_blocking=True),
            "text_padding_mask": batch["text_padding_mask"].to(device, non_blocking=True),
        }
    return {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
    }


def _trim_padding(
    logits: torch.Tensor,
    mask: torch.Tensor,
    pad_amounts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Remove zero-padding added by dataset._pad_to_divisible.
    pad_amounts: (B, 3) or (3,) — (pad_D, pad_H, pad_W).
    If all zeros (resize mode), returns tensors unchanged.
    """
    # Use first sample's pad_amounts (all samples in batch have same amounts
    # when spatial_mode="fixed" with consistent source sizes)
    if pad_amounts.dim() == 2:
        pa = pad_amounts[0]
    else:
        pa = pad_amounts

    pad_D, pad_H, pad_W = pa[0].item(), pa[1].item(), pa[2].item()
    if pad_D == 0 and pad_H == 0 and pad_W == 0:
        return logits, mask

    D = logits.shape[2] - pad_D if pad_D > 0 else logits.shape[2]
    H = logits.shape[3] - pad_H if pad_H > 0 else logits.shape[3]
    W = logits.shape[4] - pad_W if pad_W > 0 else logits.shape[4]

    logits = logits[:, :, :D, :H, :W]
    mask = mask[:, :, :D, :H, :W]
    return logits, mask
