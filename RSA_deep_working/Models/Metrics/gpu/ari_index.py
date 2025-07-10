# Metrics/gpu/ari_index.py

import torch
from torchmetrics.clustering import AdjustedRandScore

from ..base import BaseMetric


class ARIScore(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Adjusted Rand Score (ARS):
        - ARS = 1 means the prediction matches the mask perfectly.
        - ARS = 0 means the prediction is completely random.
        - ARS < 0 means the prediction is worse than random.
        """
        if abs(new_score - 1) <= abs(old_score - 1):
            return True
        return False

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        r"""Compute `Adjusted Rand Score`_ (also known as Adjusted Rand Index).

        .. math::
            ARS(U, V) = (\text{RS} - \text{Expected RS}) / (\text{Max RS} - \text{Expected RS})

        The adjusted rand score :math:`\text{ARS}` is in essence the :math:`\text{RS}` (rand score) adjusted for chance.
        The score ensures that completely randomly cluster labels have a score close to zero and only a perfect match will
        have a score of 1 (up to a permutation of the labels). The adjusted rand score is symmetric, therefore swapping
        :math:`U` and :math:`V` yields the same adjusted rand score.

        This clustering metric is an extrinsic measure, because it requires ground truth clustering labels, which may not
        be available in practice since clustering is generally used for unsupervised learning.
        """
        pred = prediction.flatten().long()
        mk = mask.flatten().long()

        ARI = AdjustedRandScore()
        return float(ARI(preds=pred, target=mk).item())
