# Metrics/cpu/euler_charac_difference.py

import numpy as np
import torch
from skimage.measure import euler_number

from ..base import BaseMetric


class EulerCharacDifference(BaseMetric):
    type = "cpu"

    def __init__(self):
        super().__init__()

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Difference de la caractéristique d’Euler :
        euler_diff = |E_pred - E_mask| / (E_pred + E_mask + ε), on renvoie 1 - mean(euler_diff).
        """
        pred_np = prediction.numpy().astype(np.uint8)
        mask_np = mask.numpy().astype(np.uint8)
        batch_size = pred_np.shape[0]
        scores = []
        for i in range(batch_size):
            pred_img = pred_np[i]
            mask_img = mask_np[i]
            e_pred = euler_number(pred_img, connectivity=1)
            e_mask = euler_number(mask_img, connectivity=1)
            diff = abs(e_pred - e_mask) / (abs(e_pred) + abs(e_mask) + 1e-8)
            scores.append(1.0 - diff)
        return float(np.mean(scores))
