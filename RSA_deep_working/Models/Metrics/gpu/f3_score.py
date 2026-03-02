# Metrics/gpu/f3_score.py

import torch
import torchmetrics.functional as FMF

from ..base import BaseMetric


class F3Score(BaseMetric):
    type = "gpu"

    def __init__(self, beta: float = 3.0, threshold: float = 0.5):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = prediction.long()
        msk = mask.long()

        score = FMF.fbeta_score(pred, msk, beta=self.beta, average="macro", task="binary")
        return score.mean().item()