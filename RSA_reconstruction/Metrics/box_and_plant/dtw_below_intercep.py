import numpy as np
from fastdtw import fastdtw
from openalea.mtg import MTG
from scipy.spatial.distance import euclidean
from utils.mtg_operations import intercept_curve

from ..base import BaseMetric


class DTWBetweenIntercepts(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """Plus le score est petit, meilleur c'est."""
        return new_score < old_score

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        x_gt, y_gt = intercept_curve(mtg_gt)
        x_pred, y_pred = intercept_curve(mtg_pred)

        if x_gt.shape != x_pred.shape:
            raise ValueError("Les courbes d'interception doivent avoir la même forme.")

        distance, _ = fastdtw(y_gt, y_pred)

        if (float(distance) != 0):
            print(f"DTW distance: {distance}")
        return float(distance)
