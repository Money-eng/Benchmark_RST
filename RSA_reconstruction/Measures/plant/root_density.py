from openalea.mtg import MTG
from utils.mtg_operations import total_root_length
from .area_convex_hull import convex_hull_area

from ..base import BaseMeasure


class RootDensity(BaseMeasure):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def __call__(self, mtg: MTG) -> float:
        total_root_lengths = total_root_length(mtg)
        
        geometry = mtg.property('geometry')  # {1: [[598.0, 148.0], [597.0, 162.0], ...]}
        points = [point for points in geometry.values() for point in points]
        ch_area = convex_hull_area(points)
        if ch_area == 0:
            return float('inf')
        
        root_density = total_root_lengths / ch_area if ch_area > 0 else float('inf')
        return root_density
