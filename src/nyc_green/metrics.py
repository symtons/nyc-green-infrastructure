"""Step 6 — Segmentation metrics: per-class IoU, accuracy, confusion matrix."""
from __future__ import annotations

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulates a confusion matrix over batches and computes metrics.

    Why a confusion matrix? Because per-class IoU is the primary metric, and
    IoU = TP / (TP + FP + FN), which is computable directly from the matrix.
    Overall accuracy is just the trace / total. Everything else downstream.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Accumulate predictions into the confusion matrix.

        logits: (B, C, H, W)
        targets: (B, H, W)
        """
        preds = logits.argmax(dim=1).cpu().numpy().ravel()
        tgts = targets.cpu().numpy().ravel()

        valid = tgts != self.ignore_index
        preds = preds[valid]
        tgts = tgts[valid]

        # Efficient confusion matrix update
        idx = self.num_classes * tgts + preds
        binned = np.bincount(idx, minlength=self.num_classes * self.num_classes)
        self.confusion += binned.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        """Return a dict of metrics computed from the accumulated confusion matrix."""
        cm = self.confusion.astype(np.float64)

        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        # IoU per class = TP / (TP + FP + FN), protecting against zero denominator
        denom = tp + fp + fn
        iou_per_class = np.where(denom > 0, tp / np.maximum(denom, 1), 0.0)

        # Overall pixel accuracy
        total = cm.sum()
        accuracy = (tp.sum() / total) if total > 0 else 0.0

        # Mean IoU (macro, over classes that actually appeared)
        present = denom > 0
        mean_iou = iou_per_class[present].mean() if present.any() else 0.0

        return {
            "accuracy": float(accuracy),
            "mean_iou": float(mean_iou),
            "iou_per_class": iou_per_class.tolist(),
            "confusion_matrix": cm.astype(np.int64).tolist(),
        }