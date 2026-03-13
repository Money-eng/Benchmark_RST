# Metrics/gpu/dice.py

import torch
import torchmetrics.functional.segmentation as FMS

from ..base import BaseMetric

class Dice(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = prediction.float()
        msk = mask.int()

        score = FMS.dice(pred, msk, ignore_index=0)
        return score.mean().item()
