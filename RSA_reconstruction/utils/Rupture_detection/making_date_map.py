
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import torch
import tifffile as tiff

from rupture_detection import RuptureDownDetector
from slope_detection import MaxSlopeDetector


@dataclass
class PredictionValidator:
    """
    Valide et normalise l'entrée: torch.Tensor (T, 1, H, W) avec valeurs dans [0,1].
    - strict: si True, lève une erreur si valeurs hors [0,1] ou NaN/Inf.
              si False, clip dans [0,1] et remplace NaN/Inf par 0.
    """
    strict: bool = True

    def __call__(self, prediction: torch.Tensor) -> np.ndarray:
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("`prediction` doit être un torch.Tensor")

        if prediction.ndim != 4:
            raise ValueError(f"`prediction` doit être de forme (T, 1, H, W), reçu {tuple(prediction.shape)}")

        T, C, H, W = prediction.shape
        if C != 1:
            raise ValueError(f"Le canal doit être 1, reçu C={C}")

        pred = prediction.detach().cpu().float()

        if self.strict:
            if not torch.isfinite(pred).all():
                raise ValueError("`prediction` contient des NaN/Inf (strict=True).")
            if (pred.min() < 0.0) or (pred.max() > 1.0):
                mn = float(pred.min())
                mx = float(pred.max())
                raise ValueError(f"Valeurs hors [0,1] détectées (min={mn}, max={mx}) (strict=True).")
        else:
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

        # (T, 1, H, W) -> (T, H, W)
        pred_np = pred[:, 0, :, :].numpy()  # float32
        return pred_np  # (T, H, W), float32 dans [0,1]

@dataclass
class ChangeCombiner:
    """
    Combine les indices:
      output_index = slope_index si rupture détectée (rupture_index != 0), sinon 0.
    Option one_based pour produire un date_map en 1..T (0 = absence).
    """
    one_based: bool = True
    max_T: int | None = None  # optionnel: utilisé pour clipper si on exporte en uint8

    def __call__(self, rupture_index: np.ndarray, slope_index: np.ndarray) -> np.ndarray:
        if rupture_index.shape != slope_index.shape:
            raise ValueError("`rupture_index` et `slope_index` doivent avoir la même forme (H, W).")

        out = np.where(rupture_index != 0, slope_index, 0).astype(np.int32)  # (H, W)

        if self.one_based:
            out = np.where(out > 0, out + 1, 0)

        if self.max_T is not None:
            # si on veut garantir la compatibilité uint8
            out = np.clip(out, 0, self.max_T).astype(np.int32)

        return out  # int (H, W)


# ------------------------------
# 5) Orchestrateur haut-niveau
# ------------------------------

@dataclass
class RuptureSlopeTimeDetector:
    """
    Chaîne complète: validation -> rupture -> pente -> combinaison.
    Utilisation:
        detector = RuptureSlopeTimeDetector(threshold_rupture=0.3, threshold_slope=0.3, one_based=True)
        date_map = detector(prediction_torch)  # prediction_torch: (T, 1, H, W), float in [0,1]
    """
    threshold_rupture: float = 0.3
    threshold_slope: float = 0.3
    one_based: bool = True
    strict_input: bool = False

    def __post_init__(self):
        self._validator = PredictionValidator(strict=self.strict_input)
        self._rupt = RuptureDownDetector(threshold_rupture=self.threshold_rupture)
        self._slope = MaxSlopeDetector(threshold_slope=self.threshold_slope)
        self._comb = ChangeCombiner(one_based=self.one_based)

    def __call__(self, prediction: torch.Tensor) -> np.ndarray:
        seq = self._validator(prediction)            # (T, H, W), float32 in [0,1]
        r_idx, _ = self._rupt(seq)                   # (H, W)
        s_idx, _ = self._slope(seq)                  # (H, W)
        date_map = self._comb(r_idx, s_idx)          # (H, W), int
        return date_map