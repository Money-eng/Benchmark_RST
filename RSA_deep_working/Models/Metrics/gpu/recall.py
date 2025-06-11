# Metrics/gpu/recall.py

import torch
import torchmetrics.functional as FMF

from ..base import BaseMetric


class Recall(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Recall binaire : TP / (TP + FN).
        """

        pred_bin = (prediction > 0.5).float()
        msk_bin = (mask > 0.5).float()

        score = FMF.recall(pred_bin, msk_bin, task="binary")
        return score.mean().item()
