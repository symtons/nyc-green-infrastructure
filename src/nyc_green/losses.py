"""Step 6 — Focal loss for imbalanced multi-class segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss with per-class alpha weighting.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: per-class weights, shape (num_classes,). Higher = emphasize that class.
        gamma: focusing parameter. Higher = focus more on hard examples. 2.0 is standard.
        ignore_index: mask values equal to this are excluded from the loss.
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: float = 2.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W)   targets: (B, H, W)
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        # Gather per-pixel class probabilities for the true class
        # pt = exp(-ce) because ce = -log(pt)
        pt = torch.exp(-ce)

        # Per-pixel alpha: look up the alpha for the true class at each pixel.
        # For ignored pixels (which will be 255), the CE is already 0, so alpha doesn't matter
        safe_targets = targets.clone()
        safe_targets[safe_targets == self.ignore_index] = 0
        alpha_per_pixel = self.alpha[safe_targets]

        focal = alpha_per_pixel * (1 - pt) ** self.gamma * ce

        # Mean over valid pixels only
        valid = (targets != self.ignore_index)
        if valid.sum() == 0:
            return torch.zeros([], device=logits.device, requires_grad=True)
        return focal[valid].mean()