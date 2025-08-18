
from typing import Tuple
import numpy as np
from attr import dataclass

@dataclass
class MaxSlopeDetector:
    """
    Détecteur de 'pente maximale' entre instants consécutifs.
    S_t = |X_t - X_{t-1}| pour t>=1 ; S_0 = 0.
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

        DIFF = np.zeros((T, P), dtype=np.float32)
        DIFF[1:] = np.abs(X[1:] - X[:-1])  # S[1:] = |Δ|
        # S[0] déjà zéro

        s_max = DIFF.max(axis=0)                 # (P,)
        k = DIFF.argmax(axis=0).astype(np.int32)  # 0..T-1

        k = np.where(s_max > self.threshold_slope, k, -1).astype(np.float32).reshape(H,W)
        #k = np.where(seq[0] > 0.5, 1, k).astype(np.int32)

        #import matplotlib.pyplot as plt
        #plt.figure(figsize=(10, 5))
        #plt.imshow(k, cmap='jet', interpolation='nearest')
        #plt.colorbar()
        #plt.title("Max Slope Index")
        #plt.show()

        return k