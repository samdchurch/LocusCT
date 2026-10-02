import torch
import torch.nn.functional as F


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
    from_logits: bool = True,
) -> torch.Tensor:
    """Per-sample Dice score. Returns shape (B,)."""
    if from_logits:
        pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    # Flatten spatial dims: (B, 1, D, H, W) -> (B, -1)
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    return (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
    from_logits: bool = True,
) -> torch.Tensor:
    """Per-sample Jaccard/IoU. Returns shape (B,)."""
    if from_logits:
        pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
    from_logits: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample precision and recall. Returns (precision (B,), recall (B,))."""
    if from_logits:
        pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    tp = (pred * target).sum(dim=1)
    precision = (tp + smooth) / (pred.sum(dim=1) + smooth)
    recall = (tp + smooth) / (target.sum(dim=1) + smooth)
    return precision, recall
