# Metrics/cpu/ari_index.py
from openalea.mtg import MTG
from util.intercept import intercept_curve_at_all_time
from ..base import BaseMetric


class TreeMatching(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def __call__(self, mtg_pred:MTG, mtg_gt:MTG):
        max_scale = mtg_gt.max_scale()
        sub_mtg_pred = []
        sub_mtg_gt = []
        for v in list(mtg_gt.vertices(scale=max_scale)):
            lengths, intercepts = intercept_curve_at_all_time(mtg_gt.sub_mtg(v), v)
            sub_mtg_gt.append((lengths, intercepts))
        for v in list(mtg_pred.vertices(scale=max_scale)):
            lengths, intercepts = intercept_curve_at_all_time(mtg_pred.sub_mtg(v), v)
            sub_mtg_pred.append((lengths, intercepts))
            
        # find the area below the intercepts for both predicted and ground truth MTGs
        area_pred = sum(lengths * intercepts for lengths, intercepts in sub_mtg_pred)
        area_gt = sum(lengths * intercepts for lengths, intercepts in sub_mtg_gt)
        # compute the area below the intercepts
        return area_pred / area_gt if area_gt != 0 else 0.0
        
