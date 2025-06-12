# Metrics/gpu/surface_difference.py

import torch
from monai.metrics.surface_distance import SurfaceDistanceMetric as MonaiSurfaceDistanceMetric

from ..base import BaseMetric

class Surface_distance(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """ 
        Compute the surface distance between the predicted segmentation and the ground truth mask.
            SD(pred, mask) = 1/δpred SUM_(border pred : p)( min_(border mask : q) || p - q ||_2 )
        In other words, it computes the sum of the minimum Euclidean distances from each point on the border of the predicted segmentation to the nearest point on the border of the ground truth mask, averaged over all batch elements.
        """
        pred = (prediction > 0.5).float()
        msk = (mask > 0.5).float()
        #  It must be one-hot format and first dim is batch.
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
            msk = msk.squeeze(1)
            
        # Compute surface distance using MONAI's SurfaceDistanceMetric
        metric = MonaiSurfaceDistanceMetric(include_background=False, distance_metric="euclidean") # define the metric
        metric(pred, msk) # compute the metric on the prediction and mask tensors
        surface_distance = metric.aggregate().item() # aggregate the results to get a single value by mean on the batch
        return surface_distance