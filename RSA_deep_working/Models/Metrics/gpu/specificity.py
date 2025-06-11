# Metrics/gpu/specificity.py

import torch

from ..base import BaseMetric


class Specificity(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Spécificité (TN / (TN + FP)) pour segmentation binaire, 
        calculée manuellement quand torchmetrics.stat_scores n'existe pas.

        On suppose que `prediction` et `mask` sont déjà des tenseurs {0,1}
        de forme [B, 1, H, W] ou [B, H, W].
        """
        pred = (prediction > 0.5).float()
        msk = (mask > 0.5).float()

        # 1) On s'assure qu'on ait un tenseur [B, H, W], pas [B,1,H,W]
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
            msk = msk.squeeze(1)

        # 2) Binarisation explicite au cas où ce seraient des probabilités :
        #    (décommenter la ligne ci-dessous si vos prédictions sont dans [0,1] et non déjà {0,1})
        # pred = (pred >= 0.5).float()
        # msk  = (msk >= 0.5).float()

        # 3) Calcul manuel de TN et FP par image :
        #    - True Negative (TN) : prédiction = 0 ET mask = 0
        #    - False Positive (FP): prédiction = 1 ET mask = 0
        #
        #    On obtient deux tenseurs de forme [B], qui comptent le nombre de pixels correspondants
        #
        #    TN_i = ((pred[i] == 0) & (msk[i] == 0)).sum()
        #    FP_i = ((pred[i] == 1) & (msk[i] == 0)).sum()

        # On crée un mask booléen pour (mask == 0) (i.e. les pixels négatifs)
        neg_mask = (msk == 0.0)

        # TN : parmi les pixels où mask=0, combien ont pred=0 ?
        tn_per_image = ((pred == 0.0) & neg_mask).view(pred.size(0), -1).sum(dim=1).float()

        # FP : parmi les pixels où mask=0, combien ont pred=1 ?
        fp_per_image = ((pred == 1.0) & neg_mask).view(pred.size(0), -1).sum(dim=1).float()

        # 4) Spécificité par image : TN / (TN + FP + ε)
        eps = 1e-8
        spec_per_image = tn_per_image / (tn_per_image + fp_per_image + eps)

        # 5) On renvoie la moyenne sur le batch
        return spec_per_image.mean().item()
