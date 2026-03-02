# Metrics/gpu/iou.py

import torch

from ..base import BaseMetric


class IoU(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = prediction.long()
        msk = mask.long()
        
        # Compute binary Jaccard (IoU)
        intersection = torch.logical_and(pred, msk).sum().item()
        union = torch.logical_or(pred, msk).sum().item()
        score = intersection / union if union > 0 else 1.0  

        return score.mean().item() if isinstance(score, torch.Tensor) else float(score)
