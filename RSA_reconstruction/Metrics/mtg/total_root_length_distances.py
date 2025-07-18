# Metrics/cpu/ari_index.py
from openalea.mtg import MTG

from ..base import BaseMetric


class TRLdistance(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Ratio of predicted to ground truth number of organs.
        On considère que `old_score` et `new_score`
        sont des scores de type float.
        """
        return abs(new_score - 1) <= abs(old_score - 1)

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        root_scale = mtg_gt.max_scale()  # ASSUMING order 2 max for roots
        verts_gt = list(mtg_gt.vertices(scale=root_scale))
        verts_pred = list(mtg_pred.vertices(scale=root_scale))

        for vertex in verts_gt:
            # get root geometry
            root_gt = mtg_gt.properties("geometry")[vertex]
            # sum of euclidean distances between all points in the root
            total_length_gt = sum(
                ((root_gt[i][0] - root_gt[i - 1][0]) ** 2 + (root_gt[i][1] - root_gt[i - 1][1]) ** 2) ** 0.5
                for i in range(1, len(root_gt))
            )
            
        for vertex in verts_pred:
            # get root geometry
            root_pred = mtg_pred.properties("geometry")[vertex]
            # sum of euclidean distances between all points in the root
            total_length_pred = sum(
                ((root_pred[i][0] - root_pred[i - 1][0]) ** 2 + (root_pred[i][1] - root_pred[i - 1][1]) ** 2) ** 0.5
                for i in range(1, len(root_pred))
            )
        # Calculate the ratio of predicted to ground truth total root lengths
        if total_length_gt == 0:
            return 0.0
        ratio = total_length_pred / total_length_gt
        
        return ratio if ratio > 0 else 0.0 
        
