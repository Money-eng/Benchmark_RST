# Metrics/cpu/euler_charac_difference.py

import numpy as np
from skimage.measure import euler_number

from ..base import BaseMetric


class EulerCharaJaccardsRatio(BaseMetric):
    type = "cpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    def __call__(self, prediction: np.ndarray, mask: np.ndarray) -> float:
        pred_np = prediction.astype(np.uint8)
        mask_np = mask.astype(np.uint8)
        batch_size = pred_np.shape[0]
        scores = []
        for i in range(batch_size):
            pred_img = pred_np[i]
            mask_img = mask_np[i]
            e_pred = euler_number(pred_img, connectivity=1)  # number of connected components - number of holes
            e_mask = euler_number(mask_img, connectivity=1)
            jaccard_ratio = (min(abs(e_pred), abs(e_mask)) / (max(abs(e_pred), abs(e_mask)) + 1e-8))
            scores.append(jaccard_ratio)
        return float(np.mean(scores))
