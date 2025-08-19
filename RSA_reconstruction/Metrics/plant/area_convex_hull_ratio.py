import numpy as np
from Measures.plant.area_convex_hull import Convex_Area_Hull
from openalea.mtg import MTG

from ..base import BaseMetric


class Area_convex_HullRatio(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score < old_score

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        return Convex_Area_Hull()(mtg_pred) / Convex_Area_Hull()(mtg_gt) if Convex_Area_Hull()(mtg_gt) > 0 else 0.0
