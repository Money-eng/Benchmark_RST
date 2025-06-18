# Metrics/gpu/pixel_accuracy.py

import torch
import torchmetrics.functional as FMF

from ..base import BaseMetric


class PixelAccuracy(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()
        
    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Pixel accuracy (binaire). On considère que `old_score` et `new_score`
        sont des scores de type float.
        """
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Pixel accuracy (binaire) : proportion de pixels correctement classés.
        """
        pred = prediction.float()
        msk = mask.float()

        pred_bin = (pred >= 0.5).long()  # TODO : check if this is the right way to binarize
        msk_bin = (msk >= 0.5).long()

        score = FMF.accuracy(pred_bin, msk_bin, task="binary")
        return score.mean().item()
