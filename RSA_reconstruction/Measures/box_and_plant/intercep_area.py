# Metrics/cpu/ari_index.py
import numpy as np
from openalea.mtg import MTG
from .intercep import Intercept_curve

from ..base import BaseMeasure
    
class Intercept_curve_Area(BaseMeasure):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()
        
    @staticmethod
    def _area_between_surfaces(x: np.ndarray,
                               y1: np.ndarray,
                               y2: np.ndarray) -> float:
        delta = np.abs(y1 - y2)

        area = np.trapz(delta, x)
        return float(area)

    def __call__(self, mtg: MTG) -> tuple:
        curve = Intercept_curve()(mtg)
        if curve[0].shape != curve[1].shape:
            raise ValueError("Les courbes d'interception doivent avoir la même forme.")
        area = self._area_between_surfaces(curve[0], curve[1], np.zeros_like(curve[1]))
        return area
        
        