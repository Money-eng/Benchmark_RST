# Metrics/cpu/ari_index.py
from openalea.mtg import MTG

from ..base import BaseMetric


class TreeMatching(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG, **kwargs):
        """
        Compute the tree matching between two MTGs.
        :param mtg_pred: predicted MTG
        :param mtg_gt: ground truth MTG
        :return: tree matching score
        """
        return 0
