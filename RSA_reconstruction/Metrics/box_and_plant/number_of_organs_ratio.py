from Measures.box_and_plant.number_of_organs import NumberOfOrgans
from openalea.mtg import MTG

from ..base import BaseMetric


class NumberOfOrgansRatio(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return abs(new_score - 1) <= abs(old_score - 1)

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        return NumberOfOrgans()(mtg_pred) / NumberOfOrgans()(mtg_gt) if NumberOfOrgans()(mtg_gt) > 0 else 0.0
