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
            VI(X; Y ) = - \sum_{i,j} r_{ij} \left[\log(r_{ij}/p_i)+\log(r_{ij}/q_j)
        where r_{ij} is the intersection of the two distributions (divided by the total number of samples),
        p_i is the marginal distribution of X, and q_j is the marginal distribution of Y.
        """
        pred_np = prediction.numpy().flatten()
        mask_np = mask.numpy().flatten()
        h_mask = entropy(mask_np)
        h_pred = entropy(pred_np)
        mi = mutual_info_score(mask_np, pred_np)
        vi = h_mask + h_pred - 2 * mi
        return float(1.0 / (1.0 + vi + 1e-8))
