# Metrics/cpu/betti0_difference.py

import numpy as np
import torch
from skimage.measure import label

from ..base import BaseMetric

class Betti0VariationIndex(BaseMetric):
    type = "cpu"

    def __init__(self):
        super().__init__()

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """ 
        Variation of Euler Characteristic (VEC) :
        VEC = (E(pred) - E(mask)) / (E(pred) + E(mask))
        where E is the Betti-0 characteristic (number of connected components).
        
        - VEC = 0 means the prediction matches the mask perfectly.
        - VEC > 0 means the prediction has more connected components or holes than the mask.
        - VEC < 0 means the prediction has fewer connected components or holes than the mask.
        """
        pred_np = prediction.numpy().astype(np.uint8)
        mask_np = mask.numpy().astype(np.uint8)
        scores = []
        for i in range(pred_np.shape[0]):
            pred_img = pred_np[i]
            mask_img = mask_np[i]
            n_pred = label(pred_img).max() # label connected regions of an integer array.
            n_mask = label(mask_img).max() # .max gives us the biggest label index, which corresponds to the number of connected components.
            ratio = ((n_pred - n_mask) / (n_mask + n_pred)) if n_mask > 0 else (1 if n_pred == 0 else 0)
            scores.append(ratio)
        return float(np.mean(scores))
