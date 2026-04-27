"""
focal_loss.py
Focal Loss for addressing severe class imbalance in CICIDS2017.

Reference:
    Lin, T.-Y., et al. (2017). Focal loss for dense object detection.
    ICCV 2017. https://arxiv.org/abs/1708.02002

Rationale:
    CICIDS2017 is heavily imbalanced (~78% BENIGN). Standard cross-entropy
    under-penalises hard minority-class examples (e.g., Heartbleed, Infiltration).
    Focal Loss down-weights easy examples via (1 - p_t)^gamma, forcing the
    model to focus on hard-to-classify minority threats.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Args:
        gamma  : Focusing parameter. gamma=0 reduces to cross-entropy.
        alpha  : Optional per-class weight tensor of shape (num_classes,).
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # (num_classes,) or None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (batch, num_classes) — raw model output (pre-softmax)
            targets : (batch,) — integer class labels
        Returns:
            scalar loss
        """
        # Cross-entropy gives log(p_t) per sample
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha,
                                  reduction="none")  # (batch,)

        # p_t = exp(-ce_loss) = probability of the correct class
        p_t = torch.exp(-ce_loss)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
