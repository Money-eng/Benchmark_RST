
from typing import Tuple
import numpy as np
from attr import dataclass

@dataclass
class MaxSlopeDetector:
    """
    Détecteur de 'pente maximale' entre instants consécutifs.
    S_t = |X_t - X_{t-1}| pour t>=1 ; S_0 = 0.
    Renvoie:
      - slope_index: argmax_t S_t (0..T-1), ou 0 si s_max <= threshold_slope
      - slope_score: s_max
    """
    threshold_slope: float = 0.5

    def __call__(self, seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if seq.ndim != 3:
            raise ValueError(f"`seq` doit être de forme (T, H, W), reçu {seq.shape}")
        T, H, W = seq.shape
        P = H * W

        if T < 2:
            slope_score = np.zeros((H, W), dtype=np.float32)
            slope_index = np.zeros((H, W), dtype=np.int32)
            return slope_index, slope_score

        X = seq.reshape(T, P)  # (T, H * W)

        S = np.zeros((T, P), dtype=np.float32)
        np.abs(X[1:] - X[:-1], out=S[1:])  # S[1:] = |Δ|
        # S[0] déjà zéro

        s_max = S.max(axis=0)                 # (P,)
        k_s = S.argmax(axis=0).astype(np.int32)  # 0..T-1

        k_s = np.where(s_max > self.threshold_slope, k_s, 0).astype(np.int32)
        return k_s.reshape(H, W), s_max.reshape(H, W).astype(np.float32)