# Metrics/cpu/ari_index.py

import numpy as np
from sklearn.metrics import adjusted_rand_score

from ..base import BaseMetric
from torchmetrics.clustering import AdjustedRandIndex

class ARIIndex(BaseMetric):
    type = "cpu"

    def __init__(self):
        super().__init__()

    def __call__(self, prediction: np.ndarray, mask: np.ndarray) -> float:
        """
        Adjusted Rand Index (ARI)
        
        ARI = 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
        
        """
        pred_np = prediction.flatten()
        mask_np = mask.flatten()

        score = adjusted_rand_score(mask_np, pred_np)
        return float(score)
