# Metrics/gpu/dice.py

import torch
import torchmetrics.functional.segmentation as FMS

from ..base import BaseMetric


class HausdorffDistance(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Hausdorff Distance (Binary).new_score est "meilleur" si elle est plus petite que old_score.
        """
        return new_score < old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Dice coefficient (Binary). On considère que `prediction` et `mask`
        sont des tenseurs de forme [B, 1, H, W] ou [B, H, W], déjà sigmoidés/binaire.
        """
        pred = prediction.long()
        msk = mask.long()

        score = FMS.hausdorff_distance(pred, msk, num_classes=2, include_background=False)
        # distance="euclidean", average="macro"
        return score.mean().item() if isinstance(score, torch.Tensor) else float(score)
