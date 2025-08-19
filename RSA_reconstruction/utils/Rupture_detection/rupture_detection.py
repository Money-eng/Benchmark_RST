from typing import Tuple

import numpy as np
from attr import dataclass


@dataclass
class RuptureDownDetector:
    """
    Détecteur de 'rupture descendante' (maximum mean-shift).
    Calcule, pour chaque pixel, l'instant i maximisant Δ(i) = mean(0..i-1) - mean(i..T-1).
    Renvoie:
      - rupture_index: i* (0..T-1, avec 0 réservé si pas de rupture => on encode i* en 1..T-1 puis remap)
      - rupture_score: Δ_max (float)
    Seuil appliqué: rupture_score > threshold_rupture  => rupture valide.
    """
    threshold_rupture: float = 0.5

    def __call__(self, seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T, H, W = seq.shape
        P = H * W

        if T < 2:
            rupture_score = np.full((H, W), -np.inf, dtype=np.float32)
            rupture_index = np.zeros((H, W), dtype=np.int32)
            return rupture_index, rupture_score

        X = seq.reshape(T, P)  # (T, P)

        csum = np.zeros((T + 1, P), dtype=np.float32)
        np.cumsum(X, axis=0, out=csum[1:])  # cumulative sum of X over time
        total = csum[T]  # (P,) # last time step c

        # Δ(i) pour i = 1..T-1
        deltas = np.empty((T - 1, P), dtype=np.float32)
        for i in range(1, T):  # time = i
            sum1 = csum[i]  # somme 0..i-1
            den1 = float(i)
            sum2 = total - csum[i]  # somme i..T-1
            den2 = float(T - i)
            mu1 = sum1 / den1
            mu2 = sum2 / den2
            deltas[i - 1] = (mu2 - mu1)

        delta_max = deltas.max(axis=0)  # (P,)
        i_star = deltas.argmax(axis=0) + 1  # indices 1..T-1

        # seuil: si Δ_max <= seuil => index 0 (pas de rupture)
        rupture_index = np.where(
            delta_max > self.threshold_rupture, i_star, -1).astype(np.float32).reshape(H, W)

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 5))
        # plt.imshow(rupture_index, cmap='jet', interpolation='nearest')
        # plt.colorbar()
        # plt.title("Rupture Index")
        # plt.show()

        return rupture_index
