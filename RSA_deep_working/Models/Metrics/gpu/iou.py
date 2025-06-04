# Metrics/gpu/iou.py

import torch
import torchmetrics.functional as FMF
from ..base import BaseMetric


class IoU(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Intersection over Union (Jaccard) for binary masks.
        We threshold both prediction and mask at 0.5, so that jaccard_index
        only sees exact 0/1 values.
        """
        # Ensure float
        pred = prediction.float()
        msk  = mask.float()

        # Binarize: anything ≥0.5 → 1, anything <0.5 → 0
        pred_bin = (pred >= 0.5).long() # TODO : check if this is the right way to binarize
        msk_bin  = (msk  >= 0.5).long()

        # Compute binary Jaccard (IoU)
        score = FMF.jaccard_index(pred_bin, msk_bin, task="binary")
        return score.mean().item()
