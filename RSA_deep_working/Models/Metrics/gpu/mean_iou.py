# Metrics/gpu/iou.py

import torch
import torchmetrics.functional.segmentation as FMF

from ..base import BaseMetric


class MeanIoU(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = prediction.long()
        msk = mask.long()

        # Compute binary Jaccard (IoU)
        score = FMF.mean_iou(pred, msk, num_classes=2)

        return score.mean().item() if isinstance(score, torch.Tensor) else float(score)
