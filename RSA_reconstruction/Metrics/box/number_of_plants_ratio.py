from openalea.mtg import MTG
from Measures.box.number_of_plants import NumberOfPlants

from ..base import BaseMetric


class NumberOfPlantsRatio(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return abs(new_score - 1) <= abs(old_score - 1)

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        return NumberOfPlants()(mtg_pred) / NumberOfPlants()(mtg_gt) if NumberOfPlants()(mtg_gt) > 0 else 0.0
