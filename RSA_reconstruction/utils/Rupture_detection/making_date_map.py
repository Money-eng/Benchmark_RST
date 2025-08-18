
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
from typing import Tuple

from utils.Rupture_detection.rupture_detection import RuptureDownDetector
from utils.Rupture_detection.slope_detection import MaxSlopeDetector

@dataclass
class ChangeCombiner:
    """
    Combine les indices:
      output_index = slope_index si rupture détectée (rupture_index != 0), sinon 0.
    """
    
    def __call__(self, rupture_index: np.ndarray, slope_index: np.ndarray, seq: np.ndarray) -> np.ndarray:
        out = np.where(rupture_index != -1, slope_index + 1, 0)
        out = np.where(seq[0] > 0.5, 1, out)
        return out


# ------------------------------
# 5) Orchestrateur haut-niveau
# ------------------------------

@dataclass
class RuptureSlopeTimeDetector:
    """
    Chaîne complète: validation -> rupture -> pente -> combinaison.
    Utilisation:
        detector = RuptureSlopeTimeDetector(threshold_rupture=0.75, threshold_slope=0.3, one_based=True)
        date_map = detector(prediction_torch)  # prediction_torch: (T, 1, H, W), float in [0,1]
    """

    threshold_rupture: float = 0.55
    threshold_slope: float = threshold_rupture
    one_based: bool = True
    max_T: int | None = None

    def __post_init__(self):
        self._rupt = RuptureDownDetector(threshold_rupture=self.threshold_rupture)
        self._slope = MaxSlopeDetector(threshold_slope=self.threshold_slope)
        self._comb = ChangeCombiner()

    # Appel simple (sans UI), si besoin
    def __call__(self, prediction: torch.Tensor) -> np.ndarray:
        seq = prediction.detach().cpu().float().numpy()[:, 0, :, :]  # (T,H,W)
        rupture_idx = self._rupt(seq)
        slope_idx = self._slope(seq)
        combined = self._comb(rupture_idx, slope_idx, seq)
        return combined