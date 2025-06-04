# Metrics/cpu/connectivity.py

import numpy as np
from skimage.measure import label
import torch
from ..base import BaseMetric


class Connectivity(BaseMetric):
    type = "cpu"

    def __init__(self):
        super().__init__()

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Connectivity metric :
        Pour chaque image du batch, on calcule le nombre de composantes connexes
        dans la prédiction vs. dans le mask, puis on fait
            conn_score = 1 - |N_pred - N_mask| / max(N_pred, N_mask)
        Et on renvoie la moyenne sur le batch.
        """
        # On fait d’abord la conversion en numpy.uint8
        pred_np = prediction.detach().cpu().numpy().astype(np.uint8)
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
        batch_size = pred_np.shape[0]
        scores = []
        for i in range(batch_size):
            pred_img = pred_np[i]
            mask_img = mask_np[i]
            num_pred = label(pred_img).max()
            num_mask = label(mask_img).max()
            denom = max(num_pred, num_mask)
            if denom == 0:
                conn_score = 1.0 if num_pred == num_mask else 0.0
            else:
                conn_score = 1.0 - abs(num_pred - num_mask) / denom
            scores.append(conn_score)
        return float(np.mean(scores))
