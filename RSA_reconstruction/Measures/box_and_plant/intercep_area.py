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
    def _area_surface(x: np.ndarray,
                               y: np.ndarray) -> float:
        return np.trapz(y, x)
        

    def __call__(self, mtg: MTG) -> tuple:
        curve = Intercept_curve()(mtg)
        if curve[0].shape != curve[1].shape:
            raise ValueError("Les courbes d'interception doivent avoir la même forme.")
        area = self._area_surface(curve[0], curve[1])
        return area
