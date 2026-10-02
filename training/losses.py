from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   (B, 1, D, H, W) logits
        target: (B, 1, D, H, W) binary float
        Returns per-sample Dice loss, shape (B,).
        """
        pred = torch.sigmoid(pred)
        pred = pred.reshape(pred.size(0), -1)
        target = target.reshape(target.size(0), -1)
        intersection = (pred * target).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        return 1.0 - dice


class CombinedLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        bce_pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        pos_weight = torch.tensor([bce_pos_weight]) if bce_pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        pred:   (B, 1, D, H, W) logits
        target: (B, 1, D, H, W) binary float
        Returns (total_loss, {"dice": float, "bce": float, "per_sample": Tensor (B,)}).
        total_loss is the batch mean, used for backward(); "per_sample" is the
        (detached) per-sample total loss, e.g. for tracking which samples
        aren't improving over training.
        """
        dice_loss = self.dice(pred, target)  # (B,)
        bce_loss = self.bce(pred, target).reshape(pred.size(0), -1).mean(dim=1)  # (B,)
        per_sample = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        total = per_sample.mean()
        return total, {
            "dice": dice_loss.mean().item(),
            "bce": bce_loss.mean().item(),
            "per_sample": per_sample.detach(),
        }
