# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from utils.Rupture_detection.rupture_detection import RuptureDownDetector
from utils.Rupture_detection.slope_detection import MaxSlopeDetector


@dataclass
class ChangeCombiner:
    def __call__(self, rupture_index: np.ndarray, slope_index: np.ndarray, seq: np.ndarray) -> np.ndarray:
        out = np.where(rupture_index != -1, slope_index + 1, 0)
        out = np.where(seq[0] > 0.5, 1, out)
        return out


@dataclass
class RuptureSlopeTimeDetector:

    threshold_rupture: float = 0.75
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
