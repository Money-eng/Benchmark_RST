# Metrics/cpu/betti0_difference.py

import numpy as np
import torch
from skimage.measure import label
from ..base import BaseMetric


class Betti0Difference(BaseMetric):
    type = "cpu"

    def __init__(self):
        super().__init__()

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Betti-0 difference : différence normalisée du nombre de composantes connexes.
        On renvoie 1 - (|N_pred - N_mask| / (N_pred + N_mask + ε)).
        """
        pred_np = prediction.detach().cpu().numpy().astype(np.uint8)
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
        batch_size = pred_np.shape[0]
        scores = []
        for i in range(batch_size):
            pred_img = pred_np[i]
            mask_img = mask_np[i]
            n_pred = label(pred_img).max()
            n_mask = label(mask_img).max()
            diff = abs(n_pred - n_mask) / (n_pred + n_mask + 1e-8)
            scores.append(1.0 - diff)
        return float(np.mean(scores))
