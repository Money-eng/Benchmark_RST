# Metrics/gpu/surface_difference.py

import torch
from monai.metrics.surface_distance import SurfaceDistanceMetric as MonaiSurfaceDistanceMetric

from ..base import BaseMetric


class Surface_distance(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score < old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = prediction.float()
        msk = mask.float()

        metric = MonaiSurfaceDistanceMetric(include_background=False, distance_metric="euclidean")  # define the metric
        metric(pred, msk)  # compute the metric on the prediction and mask tensors
        surface_distance = metric.aggregate().item()  #  aggregate the results to get a single value by mean on the batch
        return surface_distance
