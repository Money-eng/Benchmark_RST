
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch

from utils.Rupture_detection.rupture_detection import RuptureDownDetector
from utils.Rupture_detection.slope_detection import MaxSlopeDetector

@dataclass
class ChangeCombiner:
    """
    Combine les indices:
      output_index = slope_index si rupture détectée (rupture_index != 0), sinon 0.
    Option one_based pour produire un date_map en 1..T (0 = absence).
    """
    one_based: bool = True
    max_T: int | None = None  # optionnel: utilisé pour clipper si on exporte en uint8

    def __call__(self, rupture_index: np.ndarray, slope_index: np.ndarray, seq: np.ndarray) -> np.ndarray:
        if rupture_index.shape != slope_index.shape:
            raise ValueError("`rupture_index` et `slope_index` doivent avoir la même forme (H, W).")
        
        seq = np.where(seq > 0.5, seq, 0)
        seq = (seq > 0).astype(np.float32)
        bin_base_image = seq[0]
        for i in range(1, seq.shape[0]):
            bin_base_image = np.multiply(bin_base_image, seq[i])

        out = np.where(rupture_index != 0, slope_index, 0).astype(np.int32) 
        out = np.where(out == 0, out, out + 1)

        out = np.where(out == 0, bin_base_image, out)

        if self.max_T is not None: # si on veut clipper à max_T
            out = np.clip(out, 0, self.max_T).astype(np.int32)
            
        return out


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
    threshold_rupture: float = 0.75
    threshold_slope: float = 0.3 # doesn't matter as long as > 0
    one_based: bool = True

    def __post_init__(self):
        self._rupt = RuptureDownDetector(threshold_rupture=self.threshold_rupture)
        self._slope = MaxSlopeDetector(threshold_slope=self.threshold_slope)
        self._comb = ChangeCombiner(one_based=self.one_based)

    def __SIMPLEcall__(self, prediction: torch.Tensor) -> np.ndarray:
        seq = prediction.detach().cpu().float()    # (T, 1, H, W)
        seq = seq[:, 0, :, :].numpy()              # (T, H, W), float32 in [0,1]

        rupture_idx, _ = self._rupt(seq)
        slope_idx, _ = self._slope(seq)
        combined = self._comb(rupture_idx, slope_idx, seq)

        return combined
    
    def __call__(self, prediction: torch.Tensor) -> np.ndarray:
        seq = prediction.detach().cpu().float()    # (T, 1, H, W)
        seq = seq[:, 0, :, :].numpy()              # (T, H, W), float32 in [0,1]

        rupture_idx, _ = self._rupt(seq)
        slope_idx, _ = self._slope(seq)
        combined = self._comb(rupture_idx, slope_idx, seq)

        return combined

    def __PLOTcall__(self, prediction: torch.Tensor) -> np.ndarray:
        """
        Affiche 3 images côte à côte (rupture / slope / combinaison) avec 2 sliders
        pour ajuster les seuils en temps réel.
        Retourne le dernier date_map calculé (à l'état final des sliders).
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button

        # Préparer la séquence (T, H, W) en numpy
        seq = prediction.detach().cpu().float()    # (T, 1, H, W)
        seq = seq[:, 0, :, :].numpy()              # (T, H, W), float32 in [0,1]

        print("Pixel max value:", seq[0].max())
        print("Pixel min value:", seq[0].min())
        print("Pixel mean value:", seq[0].mean())

        # --- Fonctions de calcul pour un seuil donné ---
        def compute_rupture_idx(thr):
            self._rupt = RuptureDownDetector(threshold_rupture=float(thr))
            r_idx, _ = self._rupt(seq)
            return r_idx

        def compute_slope_idx(thr):
            self._slope = MaxSlopeDetector(threshold_slope=float(thr))
            s_idx, _ = self._slope(seq)
            return s_idx

        def combine(r_idx, s_idx):
            self._comb = ChangeCombiner(one_based=self.one_based)
            return self._comb(r_idx, s_idx, seq)

        # --- Figure & axes ---
        plt.close('all')
        fig = plt.figure(figsize=(12, 5))
        # 3 images sur la rangée du haut
        ax_c = plt.subplot2grid((3, 3), (0, 0), colspan=3)
        # Lignes 2-3 pour les sliders + bouton
        ax_slider_r = plt.subplot2grid((3, 3), (1, 0), colspan=3)
        ax_slider_s = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        ax_reset    = plt.subplot2grid((3, 3), (2, 2))

        # Valeurs initiales
        thr_r_init = float(self.threshold_rupture)
        thr_s_init = float(self.threshold_slope)

        r_idx = compute_rupture_idx(thr_r_init)
        s_idx = compute_slope_idx(thr_s_init)
        date_map = combine(r_idx, s_idx)

        # --- Images ---
        im_c = ax_c.imshow(date_map, cmap='jet')
        ax_c.set_title("Combinaison")
        ax_c.axis('off')

        # --- Sliders ---
        # Slider rupture (-0.5..1 par pas de 0.01)
        slider_r = Slider(
            ax=ax_slider_r, label="Seuil Rupture", valmin=-0.5, valmax=1.0,
            valinit=thr_r_init, valstep=0.01
        )

        # Slider slope (0..28 par pas de 0.01)
        slider_s = Slider(
            ax=ax_slider_s, label="Seuil Slope", valmin=0.0, valmax=2,
            valinit=thr_s_init, valstep=0.01
        )

        # Bouton reset
        btn_reset = Button(ax_reset, "Reset")

        # --- Callbacks ---
        def update(_):
            # Recalcule avec les valeurs actuelles des sliders
            thr_r = float(slider_r.val)
            thr_s = float(slider_s.val)

            r = compute_rupture_idx(thr_r)
            s = compute_slope_idx(thr_s)
            c = combine(r, s)

            # Mets à jour les images
            im_c.set_data(c)

            # Mets à jour les titres
            ax_c.set_title(f"Combinaison (thr_r={thr_r:.2f}, thr_s={thr_s:.2f})")

            # Redessine
            fig.canvas.draw_idle()

        def on_reset(event):
            slider_r.reset()
            slider_s.reset()

        slider_r.on_changed(update)
        slider_s.on_changed(update)
        btn_reset.on_clicked(on_reset)

        plt.tight_layout()
        plt.show()

        # Retourne le dernier date_map en utilisant la dernière position des sliders
        # (si tu veux capturer la valeur finale après interaction, tu peux aussi
        #  conserver c dans une variable nonlocale ; ici on recalcule proprement)
        final_r = compute_rupture_idx(float(slider_r.val))
        final_s = compute_slope_idx(float(slider_s.val))
        date_map = combine(final_r, final_s)
        return date_map
