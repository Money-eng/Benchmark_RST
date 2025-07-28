import numpy as np
from fastdtw import fastdtw
from openalea.mtg import MTG
from Measures.box_and_plant.intercep import Intercept_curve

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
        x_gt, y_gt = Intercept_curve()(mtg_gt)
        x_pred, y_pred = Intercept_curve()(mtg_pred)

        if x_gt.shape != x_pred.shape:
            raise ValueError("Les courbes d'interception doivent avoir la même forme.")

        distance, _ = fastdtw(y_gt, y_pred)
        return float(distance)
