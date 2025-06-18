# Metrics/cpu/ari_index.py
from openalea.mtg import MTG
import numpy as np
from utils.intercept import intercept_curve_at_all_time
from ..base import BaseMetric


class AreaBetweenIntercepts(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()
        
    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Area between intercepts. On considère que `old_score` et `new_score`
        sont des scores de type float.
        """
        return new_score < old_score

    @staticmethod
    def area_between_curves(x1, y1, x2, y2, num=1000) -> float:
        # bornes d'intersection des domaines
        x_min = max(min(x1), min(x2))
        x_max = min(max(x1), max(x2))
        x_common = np.linspace(x_min, x_max, num=num)
        y1i = np.interp(x_common, x1, y1)
        y2i = np.interp(x_common, x2, y2)
        return np.trapz(np.abs(y1i - y2i), x_common)

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        plant_scale = 1
        verts_gt = list(mtg_gt.vertices(scale=plant_scale))
        verts_pred = list(mtg_pred.vertices(scale=plant_scale))
        
        map_subtree_gt = {}
        map_subtree_pred = {}
        for v in verts_gt:
            map_subtree_gt[v] = mtg_gt.sub_mtg(v)
        for v in verts_pred:
            map_subtree_pred[v] = mtg_pred.sub_mtg(v)

        map_curve_gt = {}
        for v in verts_gt:
            x_gt, y_gt = intercept_curve_at_all_time(map_subtree_gt[v], 0)
            map_curve_gt[v] = (x_gt, y_gt)

        map_curve_pred = {}
        for v in verts_pred:
            x_pred, y_pred = intercept_curve_at_all_time(map_subtree_pred[v], 0)
            map_curve_pred[v] = (x_pred, y_pred)

        distance_matrix = np.zeros((len(verts_gt), len(verts_pred))) # line = GT, column = PRED
        for i, v_gt in enumerate(verts_gt):
            x_gt, y_gt = map_curve_gt[v_gt]
            for j, v_pred in enumerate(verts_pred):
                x_pred, y_pred = map_curve_pred[v_pred]
                area = 0
                
                for t in range(y_gt.shape[0]): # 29 times
                    area += self.area_between_curves(x_gt, y_gt[t, :], x_pred, y_pred[t, :])
                distance_matrix[i, j] = area
        
        # for each line in the distance matrix, find the minimum value
        min_values = np.min(distance_matrix, axis=1)
        # return the sum the minimum values
        return np.sum(min_values)