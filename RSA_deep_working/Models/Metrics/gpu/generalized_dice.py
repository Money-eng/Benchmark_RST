# Metrics/gpu/dice.py

import torch
import torchmetrics.functional.segmentation as FMS

from ..base import BaseMetric


class GeneralizedDice(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Dice coefficient (Binary). On considère que `old_score` et `new_score`
        sont des scores de type float.
        """
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Dice coefficient (Binary). On considère que `prediction` et `mask`
        sont des tenseurs de forme [B, 1, H, W] ou [B, H, W], déjà sigmoidés/binaire.
        """
        pred = prediction.float()
        msk = mask.float()

        score = FMS.generalized_dice_score(
            pred, msk, num_classes=2
        )
        return score.mean().item() if isinstance(score, torch.Tensor) else float(score)
