# Metrics/cpu/ari_index.py
from rsml.measurements import root_length

from ..base import BaseMetric


class NumberOfLateralsRatio(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return abs(new_score - 1) <= abs(old_score - 1)

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        root_scale = mtg_gt.max_scale()  # ASSUMING order 2 max for roots
        verts_gt = list(mtg_gt.vertices(scale=root_scale))
        verts_pred = list(mtg_pred.vertices(scale=root_scale))

        num_root_gt = len(verts_gt)
        num_root_pred = len(verts_pred)

        return num_root_pred / num_root_gt if num_root_gt > 0 else 0.0
