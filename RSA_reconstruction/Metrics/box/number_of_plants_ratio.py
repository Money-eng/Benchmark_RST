from openalea.mtg import MTG
from rsml.misc import plant_vertices

from ..base import BaseMetric


class NumberOfPlantsRatio(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Ratio of predicted to ground truth number of plants.
        On considère que `old_score` et `new_score`
        sont des scores de type float.
        """
        return abs(new_score - 1) <= abs(old_score - 1)

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        verts_gt = plant_vertices(mtg_gt)
        verts_pred = plant_vertices(mtg_pred)

        num_plants_gt = len(verts_gt)
        num_plants_pred = len(verts_pred)

        return num_plants_pred / num_plants_gt if num_plants_gt > 0 else 0.0
