from __future__ import annotations

from pathlib import Path
from typing import Union, List, Dict, Optional, Sequence

import napari  # pip install napari[all]
import numpy as np
import tifffile  # pip install tifffile
from magicgui import magicgui  # pip install magicgui
from matplotlib import pyplot as plt
from openalea.mtg import MTG
from rsml import rsml2mtg


class RootGraphViewer:
    _DEFAULT_COLORS: Dict[str, str] = {
        "gt_expertized": "#2ca02c",  # vert
        "gt_before_expertized": "#ff7f0e",  # orange
        "prediction": "#d62728",  # rouge
    }

    _DEFAULT_OPACITY: Dict[str, float] = {
        "prediction": 1,
        "gt_before_expertized": 1,
        "gt_expertized": 1,
    }

    def __init__(
            self,
            gt_expertized: Union[str, Path, MTG, Sequence[Union[str, Path, MTG]]],
            gt_before: Union[str, Path, MTG, Sequence[Union[str, Path, MTG]]],
            prediction: Union[str, Path, MTG, Sequence[Union[str, Path, MTG]]],
            background: Optional[Union[str, Path]] = None,
            pred_date_map: Optional[Union[str, Path]] = None,
            gt_date_map: Optional[Union[str, Path]] = None,
            colors: Optional[Dict[str, str]] = None,
            flip_axes: bool = True,
            with_time: bool = True,
    ) -> None:

        self.mtg_dict: Dict[str, MTG] = {
            "prediction": self._to_mtg(prediction),
            "gt_before_expertized": self._to_mtg(gt_before),
            "gt_expertized": self._to_mtg(gt_expertized),
        }

        self.background = Path(background).expanduser() if background else None
        self.pred_date_map = Path(pred_date_map).expanduser() if pred_date_map else None
        self.gt_date_map = Path(gt_date_map).expanduser() if gt_date_map else None
        self.colors = {**self._DEFAULT_COLORS, **(colors or {})}
        self.flip_axes = flip_axes

        self._temporal_shapes: Dict[str, List[List[np.ndarray]]] = {}
        self.with_time = with_time

    def show(self) -> napari.Viewer:
        viewer = napari.Viewer(title="Comparing Root Systems")
        shapes_layers: Dict[str, napari.layers.Shapes] = {}
        n_frames = 1

        # — background image stack —
        if self.background and self.background.exists():
            stack = tifffile.imread(str(self.background))
            viewer.add_image(stack, name="background", blending="additive")
            n_frames = stack.shape[0] if stack.ndim == 3 else 1

        # — date‑map prédiction —
        if self.pred_date_map and self.pred_date_map.exists():
            raw = tifffile.imread(str(self.pred_date_map)).astype(np.float32)
            pred_layer = viewer.add_image(
                raw,
                name="prediction_date_map",
                blending="additive",
                contrast_limits=[0, 29],
            )
            pred_layer.visible = False

        # — date‑map GT —
        if self.gt_date_map and self.gt_date_map.exists():
            raw = tifffile.imread(str(self.gt_date_map)).astype(np.float32)
            gt_layer = viewer.add_image(
                raw,
                name="gt_date_map",
                blending="additive",
                contrast_limits=[0, 29],
            )
            gt_layer.visible = False

            # — préparer les LUTs —
        # turbo pour 1–29, 0 = noir
        base = plt.get_cmap("turbo", 29)
        turbo_lut = np.zeros((30, 4), dtype=float)
        turbo_lut[1:] = base(np.arange(29))
        turbo_lut[0] = [0, 0, 0, 1]

        # rouge uni 1–29
        red_lut = np.zeros((30, 4), dtype=float)
        red_lut[1:] = [1, 0, 0, 1]
        red_lut[0] = [0, 0, 0, 1]

        # vert uni 1–29
        green_lut = np.zeros((30, 4), dtype=float)
        green_lut[1:] = [0, 1, 0, 1]
        green_lut[0] = [0, 0, 0, 1]

        # — callback pour basculer les colormaps selon visibilité —
        def update_colormaps(event=None):
            both_on = getattr(pred_layer, "visible", False) and getattr(gt_layer, "visible", False)
            if both_on:
                pred_layer.colormap = red_lut
                gt_layer.colormap = green_lut
            else:
                if getattr(pred_layer, "visible", False):
                    pred_layer.colormap = turbo_lut
                if getattr(gt_layer, "visible", False):
                    gt_layer.colormap = turbo_lut

        # connecter aux événements .visible
        pred_layer.events.visible.connect(update_colormaps)
        gt_layer.events.visible.connect(update_colormaps)
        update_colormaps()  # initialisation

        # — préparation des shapes temporelles —
        if self.with_time:
            for key, mtg in self.mtg_dict.items():
                for t in range(n_frames):
                    self._temporal_shapes.setdefault(key, []).append(
                        self._mtg_to_paths(mtg, t)
                    )

        # — ajouter les racines —
        for key, mtg in self.mtg_dict.items():
            data0 = (self._temporal_shapes[key][0]
                     if self.with_time
                     else self._mtg_to_paths(mtg))
            if not data0:
                print(f"No data found for {key}, skipping.")
                continue
            layer = viewer.add_shapes(
                data=data0,
                shape_type="path",
                edge_color=self.colors[key],
                edge_width=1,
                name=key,
                opacity=self._DEFAULT_OPACITY.get(key, 1.0),
            )
            shapes_layers[key] = layer

        # — slider temporel —
        @viewer.dims.events.current_step.connect
        def update_layers(event):
            frame, _, _ = event.value
            for key, layer in shapes_layers.items():
                if self.with_time:
                    layer.data = self._temporal_shapes[key][frame]
                else:
                    layer.data = self._mtg_to_paths(self.mtg_dict[key])

        # échelle & axes
        viewer.scale_bar.visible = True
        viewer.axes.visible = True

        napari.run()
        return viewer

    @classmethod
    def from_rsml(
            cls,
            exp_path: Union[str, Path, Sequence[Union[str, Path]]],
            before_path: Union[str, Path, Sequence[Union[str, Path]]],
            pred_path: Union[str, Path, Sequence[Union[str, Path]]],
            **kwargs,
    ) -> "RootGraphViewer":
        return cls(exp_path, before_path, pred_path, **kwargs)

    def _to_mtg(self, obj: Union[str, Path, MTG]) -> MTG:
        if hasattr(obj, "property"):
            return obj
        return rsml2mtg(str(obj))

    def _mtg_to_paths(self, mtg: MTG, time: int = -1) -> List[np.ndarray]:
        polys = mtg.property("geometry")
        paths: List[np.ndarray] = []
        for key, coords in polys.items():
            arr = np.array(coords)[:, ::-1]  # (x,y) → (row,col)
            if self.flip_axes:
                if time < 0:
                    arr = np.unique(arr, axis=0)
                else:
                    times = np.array(mtg.property("time").get(key, []))
                    arr = arr[times <= time + 1]
                    arr = np.unique(arr, axis=0)
                    if arr.shape[0] == 0:
                        continue
                    if arr.shape[0] == 1:
                        arr = np.concatenate([arr, arr[-1:]])
            paths.append(arr)
        return paths


if __name__ == "__main__":
    gt_expertized = "/home/loai/Documents/code/RSMLExtraction/RSA_deep_working/Data/Val/230629PN012/61_graph.rsml"
    gt_before = "/home/loai/Documents/code/RSMLExtraction/RSA_deep_working/Data/Val/230629PN012/61_before_expertized_graph.rsml"
    prediction = "/home/loai/Documents/code/RSMLExtraction/Results/Reconstruction_per_epoch/Segformer_bce_dice_200/Val/230629PN012/61_prediction_before_expertized_graph.rsml"
    viewer = RootGraphViewer.from_rsml(
        gt_expertized,
        gt_before,
        prediction,
        background="/home/loai/Documents/code/RSMLExtraction/RSA_deep_working/Data/Val/230629PN012/22_registered_stack.tif",
        pred_date_map="/home/loai/Documents/code/RSMLExtraction/Results/Reconstruction_per_epoch/Segformer_bce_dice_200/Val/230629PN012/40_date_map.tif",
        gt_date_map="/home/loai/Documents/code/RSMLExtraction/RSA_deep_working/Data/Val/230629PN012/40_date_map.tif",
    )

    viewer.show()