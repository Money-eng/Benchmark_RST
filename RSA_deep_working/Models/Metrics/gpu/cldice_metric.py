import numpy as np
from skimage.morphology import skeletonize
import torch

from ..base import BaseMetric

def cl_score(v, s):
    return float(np.sum(v * s)) / float(np.sum(s))

def clDice(v_p, v_l):
    tprec = cl_score(v_p, skeletonize(v_l))
    tsens = cl_score(v_l, skeletonize(v_p))
    return 2.0 * tprec * tsens / (tprec + tsens)

class CLDICE_metric(BaseMetric):
    type = "gpu"

    def __init__(self, threshold: float = 0.5, radius: float = 5.0, sampling=None):
        super().__init__()
        self.threshold = threshold
        self.radius = radius
        self.sampling = sampling

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        if torch.is_tensor(prediction):
            pred = prediction.float()
        else:
            pred = np.array(prediction, dtype=np.float32)

        if torch.is_tensor(mask):
            msk = mask.float().cpu().numpy()
        else:
            msk = np.array(mask, dtype=np.float32)
        
        cldice_scores = []
        for i in range(pred.shape[0]):
            try:
                cldice_score = clDice(pred[i], msk[i])
            except ZeroDivisionError:
                cldice_score = 0.0
            cldice_scores.append(float(cldice_score))
        return float(np.mean(cldice_scores))
