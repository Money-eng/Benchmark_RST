from openalea.mtg import MTG
from utils.mtg_operations import total_root_length

from ..base import BaseMetric


class TotalRootLengthRatio(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return abs(new_score - 1) <= abs(old_score - 1)

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        total_root_length_gt = total_root_length(mtg_gt)
        total_root_length_pred = total_root_length(mtg_pred)

        return total_root_length_pred / total_root_length_gt
