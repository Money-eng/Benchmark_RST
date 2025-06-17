# Metrics/cpu/ari_index.py
from openalea.mtg import MTG
from ..base import BaseMetric


class NumberOfOrgansRatio(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()


    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        root_scale = mtg_gt.max_scale() # ASSUMING order 2 max for roots
        verts_gt = list(mtg_gt.vertices(scale=root_scale))
        verts_pred = list(mtg_pred.vertices(scale=root_scale))
        
        num_plants_gt = len(verts_gt)
        num_plants_pred = len(verts_pred)
        
        return num_plants_pred / num_plants_gt if num_plants_gt > 0 else 0.0