import numpy as np
from openalea.mtg import MTG
from Measures.box_and_plant.intercep import Intercept_curve

from ..base import BaseMetric


class EuclidianDistancebtwIntercepts(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score < old_score

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        curve_gt = Intercept_curve()(mtg_gt)
        curve_pred = Intercept_curve()(mtg_pred)

        if curve_gt[0].shape != curve_pred[0].shape:
            raise ValueError("Les courbes d'interception doivent avoir la même forme.")

        # Calculate the Euclidean distance between the two curves
        return np.linalg.norm(curve_gt[1] - curve_pred[1])
