import numpy as np
from openalea.mtg import MTG
from scipy.spatial import ConvexHull

from ..base import BaseMetric


def convex_hull_area(points):
    """
    Calculate the area of the convex hull for a set of 2D points.
    :param points: Iterable of (x, y) pairs.
    :return: Area of the convex hull.
    """
    points = np.array(points)
    hull = ConvexHull(points)
    return hull.area  # For 2D, use hull.area; for 3D, use hull.volume


class Area_convex_Hull(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score < old_score

    def __call__(self, mtg_pred: MTG, mtg_gt: MTG) -> float:
        """Compute the area of the convex hull of the plant in the MTG."""

        pred_geometry = mtg_pred.property('geometry')  # {1: [[598.0, 148.0], [597.0, 162.0], ...]}
        gt_geometry = mtg_gt.property('geometry')

        # Extract points from the geometry
        pred_points = [point for points in pred_geometry.values() for point in points]
        gt_points = [point for points in gt_geometry.values() for point in points]

        # Compute the convex hull area for both predicted and ground truth points
        pred_area = convex_hull_area(pred_points)
        gt_area = convex_hull_area(gt_points)

        # Compute the IoU (Intersection over Union) as the metric
        if gt_area == 0:
            return float('inf')  # If ground truth area is zero, IoU is undefined

        iou = pred_area / gt_area
        return iou
