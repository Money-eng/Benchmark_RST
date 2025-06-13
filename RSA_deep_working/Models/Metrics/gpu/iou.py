# Metrics/gpu/iou.py

import torch
import torchmetrics.functional.segmentation as FMF

from ..base import BaseMetric


class MeanIoU(BaseMetric):
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
        msk = mask.float()

        # Binarize: anything ≥0.5 → 1, anything <0.5 → 0
        pred_bin = (pred >= 0.5).long()  # TODO : check if this is the right way to binarize
        msk_bin = (msk >= 0.5).long()

        # Compute binary Jaccard (IoU)
        score = FMF.mean_iou(
            pred_bin, msk_bin, num_classes=2
        )
        # `score` is a tensor, we take the mean and convert to float
        return score.mean().item() if isinstance(score, torch.Tensor) else float(score)
