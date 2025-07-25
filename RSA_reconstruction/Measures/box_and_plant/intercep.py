# Metrics/cpu/ari_index.py
import numpy as np
from openalea.mtg import MTG
from utils.mtg_operations import intercept_curve

from ..base import BaseMeasure


class Intercept_curve(BaseMeasure):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def __call__(self, mtg: MTG) -> tuple:
        return intercept_curve(mtg)  # x = array([0, 0.1, 0.2, ...,]), y = array([0, 0, 1, 1, ...])
        