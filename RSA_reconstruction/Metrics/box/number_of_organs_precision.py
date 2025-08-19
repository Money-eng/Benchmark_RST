from openalea.mtg import MTG
from rsml.matching import match_plants, match_roots

from ..base import BaseMetric


class NumberOfOrgansPrecision(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return abs(new_score - 1) <= abs(old_score - 1)

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        plant_match, _, _ = match_plants(mtg_pred, mtg_gt)
        roots, unmatched_pred, _ = match_roots(mtg_pred, mtg_gt, plant_match, max_distance=5.0)

        TP = len(roots)
        FP = len(unmatched_pred)

        return TP / (TP + FP) if (TP + FP) > 0 else 0.0
