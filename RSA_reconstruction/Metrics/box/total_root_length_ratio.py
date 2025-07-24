from openalea.mtg import MTG
from rsml.measurements import root_length, RSML_Measurements


from ..base import BaseMetric


class TotalRootLengthRatio(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return abs(new_score - 1) <= abs(old_score - 1)

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        dict_root_lenb_gt = root_length(mtg_gt) # root_indx -> root_length
        dict_root_lenb_pred = root_length(mtg_pred)
        
        total_root_length_gt = sum(dict_root_lenb_gt.values())
        total_root_length_pred = sum(dict_root_lenb_pred.values())
        
        from rsml.matching import match_plants
        matched_plants = match_plants(mtg_gt, mtg_pred)
        print(matched_plants)

        return total_root_length_pred / total_root_length_gt
