# Metrics/gpu/dice.py

import torch
from monai.metrics import compute_hausdorff_distance
from ..base import BaseMetric
import os

class HausdorffDistance95(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()
        self.pixel_size = 76 * 1e-3 # Convert pixel size to millimeters

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score < old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred_bin = prediction.float().detach().cpu()
        mask_bin = mask.float().detach().cpu()

        hd95_tensor = compute_hausdorff_distance(
            y_pred=pred_bin,
            y=mask_bin,
            include_background=False,
            distance_metric="euclidean",
            percentile=95,
            spacing=self.pixel_size
        )
        
        valid_hd95 = hd95_tensor[torch.isfinite(hd95_tensor)]
        
        if valid_hd95.numel() == 0: # if all values are inf or NaN, return 0.0 as a default value
            return 0.0
        
        return valid_hd95.mean().item()
