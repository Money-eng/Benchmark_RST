# Metrics/cpu/are_error.py

import numpy as np
import torch
from skimage.metrics import adapted_rand_error
from ..base import BaseMetric


class AREError(BaseMetric):
    type = "cpu"

    def __init__(self):
        super().__init__()

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor, time, mtg) -> float:
        """
        Adapted Rand Error pour segmentation binaire :
        ARE = 1 - ARI <– (adapted_rand_error renvoie (error, _, _)).
        """
        pred_np = prediction.numpy().astype(np.uint8)
        mask_np = mask.numpy().astype(np.uint8)
        # adapted_rand_error attend (gt, pred) en uint8
        try:
            are_value, _, _ = adapted_rand_error(mask_np, pred_np)
        except Exception:
            # En cas d’erreur (ex. image vide), on renvoie 1.0 (erreur max)
            are_value = 1.0
        return float(are_value)
