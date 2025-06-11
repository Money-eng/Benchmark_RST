# Metrics/gpu/precision.py

import torch
import torchmetrics.functional as FMF

from ..base import BaseMetric


class Precision(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Precision binaire : TP / (TP + FP).
        """
        pred_bin = (prediction > 0.5).float()
        msk_bin = (mask > 0.5).float()

        score = FMF.precision(pred_bin, msk_bin, task="binary")
        return score.mean().item()
