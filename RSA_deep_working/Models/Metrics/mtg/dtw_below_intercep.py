# Metrics/cpu/dtw_between_intercepts.py
import numpy as np
from openalea.mtg import MTG
from utils.intercept import intercept_curve_at_all_time
from fastdtw import fastdtw
from ..base import BaseMetric

class DTWBetweenIntercepts(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()
        
    def is_better(self, old_score: float, new_score: float) -> bool:    
        """
        Dynamic Time Warping (DTW) between intercepts. On considère que `old_score` et `new_score`
        sont des scores de type float.
        """
        return new_score < old_score

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
        
        num_timesteps = y_gt.shape[0]  # Should be 29
        distance_matrix = np.zeros((len(verts_gt), len(verts_pred)))
        for i, v_gt in enumerate(verts_gt):
            x_gt, y_gt = map_curve_gt[v_gt]
            for j, v_pred in enumerate(verts_pred):
                x_pred, y_pred = map_curve_pred[v_pred]
                dtw = 0
                for t in range(num_timesteps):
                    y_gt_t = y_gt[t, :]
                    y_pred_t = y_pred[t, :]
                    dtw += fastdtw(y_gt_t, y_pred_t)[0] # 0 for distance, 1 for path
                distance_matrix[i, j] = dtw
                
        # for each line in the distance matrix, find the minimum value
        min_values = np.min(distance_matrix, axis=1)
        # return the sum of the minimum values
        return np.sum(min_values)