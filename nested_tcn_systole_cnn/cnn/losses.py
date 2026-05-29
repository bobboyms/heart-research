from __future__ import annotations


import argparse
import torch
from torch import nn


class BinaryFocalWithLogitsLoss(nn.Module):
    def __init__(
        self,
        gamma: float,
        alpha: float | None = None,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "none",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.register_buffer("pos_weight", pos_weight)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        focal_factor = torch.pow((1.0 - pt).clamp_min(1e-8), self.gamma)
        loss = bce * focal_factor
        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = loss * alpha_factor
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_binary_loss(args: argparse.Namespace, pos: float, neg: float, device: torch.device) -> nn.Module:
    if getattr(args, "loss", "bce") == "focal":
        alpha = getattr(args, "focal_alpha", None)
        if alpha is None:
            total = pos + neg
            alpha = float(neg / total) if total > 0 else 0.5
        return BinaryFocalWithLogitsLoss(
            gamma=float(getattr(args, "focal_gamma", 2.0)),
            alpha=float(alpha),
            pos_weight=None,
            reduction="none",
        )
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")


def pairwise_auc_margin_loss(logits: torch.Tensor, targets: torch.Tensor, margin: float) -> torch.Tensor:
    targets = targets.to(dtype=torch.float32)
    pos_scores = logits[targets == 1]
    neg_scores = logits[targets == 0]
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return logits.sum() * 0.0
    differences = pos_scores[:, None] - neg_scores[None, :]
    return torch.relu(float(margin) - differences).mean()


def add_auc_loss(
    loss: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    auc_weight = float(getattr(args, "auc_loss_weight", 0.0))
    if auc_weight <= 0.0:
        return loss
    return loss + auc_weight * pairwise_auc_margin_loss(
        logits.reshape(-1),
        targets.reshape(-1),
        float(getattr(args, "auc_loss_margin", 1.0)),
    )
