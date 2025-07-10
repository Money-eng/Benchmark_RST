import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize_3d


class RadialDistanceLoss(nn.Module):
    """
    Radial Distance Loss pour structures tubulaires 3D.
    L = -1/2 * sum_k W_k * (2 * sum_i p_i,k * d_i,k) / (sum_i p_i,k^2 + sum_i d_i,k^2)
    où d_i,k est la carte de distance normalisée issue du squelette de la classe k.
    """

    def __init__(self, weight: torch.Tensor = None, smooth: float = 1e-5):
        """
        :param weight: tensor de taille (C,) pour pondérer chaque classe k
        :param smooth: epsilon pour stabiliser les divisions
        """
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else torch.ones(1))
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param logits: (B, C, D, H, W) sorties brutes du réseau
        :param targets: (B, C, D, H, W) one-hot des masques GT
        """
        B, C, D, H, W = logits.shape
        probs = F.softmax(logits, dim=1)
        total_loss = 0.0

        # itérer sur le batch et les classes
        for b in range(B):
            for k in range(C):
                gt_k = targets[b, k].cpu().numpy().astype(bool)

                # 1. extraire le squelette du masque GT
                skel = skeletonize_3d(gt_k)

                # 2. distance transform sur le complément du squelette
                dist = distance_transform_edt(~skel).astype(np.float32)

                # 3. normalisation [0,1]
                if dist.max() > 0:
                    dist /= dist.max()
                Dk = torch.from_numpy(dist).to(logits.device)

                pk = probs[b, k]
                # calcul du numérateur et du dénominateur
                num = 2.0 * torch.sum(pk * Dk)
                den = torch.sum(pk * pk) + torch.sum(Dk * Dk) + self.smooth

                loss_k = - (num / den)
                total_loss += self.weight[k] * loss_k

        # moyenne sur le batch et les classes
        return total_loss / (B * C)
