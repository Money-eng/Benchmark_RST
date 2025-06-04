# Metrics/cpu/vi_index.py

import numpy as np
import torch
from sklearn.metrics.cluster import entropy, mutual_info_score
from ..base import BaseMetric


class VIIndex(BaseMetric):
    type = "cpu"

    def __init__(self):
        super().__init__()

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Variation of Information (VI) :
            VI = H(mask) + H(pred) - 2 * MI(mask, pred)
        Pour avoir un score « plus haut = meilleur », on renvoie 1/(1+VI).
        """
        pred_np = prediction.detach().cpu().numpy().flatten()
        mask_np = mask.detach().cpu().numpy().flatten()
        h_mask = entropy(mask_np)
        h_pred = entropy(pred_np)
        mi = mutual_info_score(mask_np, pred_np)
        vi = h_mask + h_pred - 2 * mi
        return float(1.0 / (1.0 + vi + 1e-8))
