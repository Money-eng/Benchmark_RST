# Metrics/cpu/ari_index.py
import numpy as np
from openalea.mtg import MTG
from utils.mtg_operations import intercept_curve

from ..base import BaseMetric


class AreaBetweenIntercepts(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """Plus petit est meilleur."""
        return new_score < old_score

    @staticmethod
    def _area_between_surfaces(x: np.ndarray,
                               y1: np.ndarray,
                               y2: np.ndarray) -> float:
        delta = np.abs(y1 - y2)

        area = np.trapz(delta, x)
        return float(area)

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        curve_gt = intercept_curve(mtg_gt)  # x = array([0, 0.1, 0.2, ...,]), y = array([0, 0, 1, 1, ...])
        curve_pred = intercept_curve(mtg_pred)

        if curve_gt[0].shape != curve_pred[0].shape:
            raise ValueError("Les courbes d'interception doivent avoir la même forme.")
        area = self._area_between_surfaces(curve_gt[0], curve_gt[1], curve_pred[1])
        return area
