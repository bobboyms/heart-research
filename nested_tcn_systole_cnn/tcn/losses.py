from __future__ import annotations


import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


from .config import IGNORE_INDEX, systole_probability_index


def class_weights_from_counts(counts: np.ndarray) -> torch.Tensor:
    counts = np.maximum(counts.astype(np.float64), 1.0)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def apply_systole_weight_multiplier(
    class_weights: torch.Tensor,
    target_mode: str,
    multiplier: float,
) -> torch.Tensor:
    if multiplier <= 0:
        raise ValueError("--systole-weight-multiplier must be greater than 0.")
    weights = class_weights.clone()
    weights[systole_probability_index(target_mode)] *= float(multiplier)
    return weights


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        mode: str,
        class_weights: torch.Tensor | None,
        num_classes: int,
        dice_weight: float,
        focal_gamma: float,
        label_smoothing: float,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=IGNORE_INDEX,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        valid = target != IGNORE_INDEX
        if self.mode.startswith("focal"):
            pt = torch.exp(-ce).clamp(min=1e-6, max=1.0)
            ce = ((1.0 - pt) ** self.focal_gamma) * ce
        base_loss = ce[valid].mean() if valid.any() else ce.mean()

        if "dice" not in self.mode:
            return base_loss
        dice = multiclass_dice_loss(logits, target, self.num_classes)
        return base_loss + self.dice_weight * dice


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    if not valid.any():
        return logits.sum() * 0.0

    probs = torch.softmax(logits, dim=1).permute(0, 2, 1)
    safe_target = target.clamp_min(0)
    one_hot = F.one_hot(safe_target, num_classes=num_classes).to(dtype=probs.dtype)
    valid_f = valid.unsqueeze(-1).to(dtype=probs.dtype)
    probs = probs * valid_f
    one_hot = one_hot * valid_f

    intersection = (probs * one_hot).sum(dim=(0, 1))
    denominator = probs.sum(dim=(0, 1)) + one_hot.sum(dim=(0, 1))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def create_loss(args: argparse.Namespace, class_weights: torch.Tensor | None, num_classes: int) -> SegmentationLoss:
    return SegmentationLoss(
        mode=args.loss,
        class_weights=class_weights,
        num_classes=num_classes,
        dice_weight=args.dice_weight,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )
