# Metrics/gpu/ari_index.py

import torch
from torchmetrics.clustering import NormalizedMutualInfoScore
from ..base import BaseMetric


class NormalizedMutualInformation(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred_np = prediction.long().flatten()
        mask_np = mask.long().flatten()

        normalized_mutual_info_score = NormalizedMutualInfoScore("geometric")
    
        val = normalized_mutual_info_score(pred_np, mask_np)
        return float(val.item())
