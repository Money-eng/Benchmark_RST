# Metrics/gpu/dice.py

import torch
import torchmetrics.functional.segmentation as FMS
from ..base import BaseMetric


class Dice(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor, time, mtg) -> float:
        """
        Dice coefficient (Binary). On considère que `prediction` et `mask`
        sont des tenseurs de forme [B, 1, H, W] ou [B, H, W], déjà sigmoidés/binaire.
        """
        pred = prediction.float()
        msk = mask.float()
        
        # torchmetrics.segmentation.dice_score attend (B, H, W) ou (B, C, H, W) mais ici C=1
        score = FMS.dice_score(pred, msk, num_classes=2, average="micro")
        # `score` est un tenseur, on prend la moyenne puis on transforme en float
        return score.mean().item()
