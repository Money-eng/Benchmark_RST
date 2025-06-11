# Metrics/cpu/ari_index.py

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from ..base import BaseMetric


class ARIIndex(BaseMetric):
    type = "cpu"

    def __init__(self):
        super().__init__()

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Adjusted Rand Index (ARI) sur la segmentation binaire :
        on aplatit les tenseurs et on applique sklearn.metrics.adjusted_rand_score.
        """
        pred_np = prediction.numpy().flatten()
        mask_np = mask.numpy().flatten()

        score = adjusted_rand_score(mask_np, pred_np)
        return float(score)
