from __future__ import annotations
import numpy as np
import napari
from rsml import rsml2mtg
from pathlib import Path
from openalea.mtg import MTG
from typing import Union, List, Dict, Optional


class RootGraphViewer:
    _DEFAULT_COLORS: Dict[str, str] = {
        "gt_expertized": "#2ca02c",   # vert
        "gt_before_expertized":     "#ff7f0e",   # orange
        "prediction":    "#d62728",   # rouge
    }
    _DEFAULT_OPACITY: Dict[str, float] = {
        "prediction": 1,
        "gt_before_expertized": 1,
        "gt_expertized": 1,
    }

    def __init__(
        self,
        gt_expertized: Union[str, Path],
        gt_before:     Union[str, Path],
        prediction:    Union[str, Path],
        background:    Optional[Union[str, Path]] = None,
        colors:        Optional[Dict[str, str]] = None,
        flip_axes:     bool = True,
    ):
        self.mtg_dict = {
            "prediction":    self._to_mtg(prediction),
            "gt_before_expertized": self._to_mtg(gt_before),
            "gt_expertized": self._to_mtg(gt_expertized),
        }
        self.background = background
        self.colors = {**self._DEFAULT_COLORS, **(colors or {})}
        self.flip_axes = flip_axes

    def show(self) -> napari.Viewer:
        viewer = napari.Viewer(title="Comparaison d'arborescences")

        # Charger fond d'image si fourni
        if self.background:
            from matplotlib import pyplot as plt
            img = plt.imread(str(self.background), format="tif")
            viewer.add_image(img, name="background", blending="additive")
            
        # Ajouter chaque arborescence comme calque de shapes
        for key, mtg in self.mtg_dict.items():
            paths = self._mtg_to_paths(mtg)
            if not paths:
                print(f"Avertissement: pas de géométrie pour {key}.")
                continue
            viewer.add_shapes(
                data=paths,
                shape_type="path",
                edge_color=self.colors[key],
                edge_width=1,
                name=key,
                opacity=self._DEFAULT_OPACITY.get(key, 1.0),
            )

        # Afficher scale bar et axes
        try:
            viewer.scale_bar.visible = True
            viewer.axes.visible = True
        except AttributeError:
            pass

        napari.run()
        return viewer

    @classmethod
    def from_rsml(
        cls,
        exp_path:   Union[str, Path],
        before_path: Union[str, Path],
        pred_path:  Union[str, Path],
        **kwargs,
    ) -> "RootGraphViewer":
        return cls(exp_path, before_path, pred_path, **kwargs)

    def _to_mtg(self, obj) -> MTG:
        if hasattr(obj, "property"):
            return obj
        return rsml2mtg(str(obj))

    def _mtg_to_paths(self, mtg) -> List[np.ndarray]:
        polys = mtg.property("geometry")
        paths: List[np.ndarray] = []
        for vid, coords in polys.items():
            arr = np.array(coords)
            if self.flip_axes:
                # Napari uses (row, col) ordering
                arr = arr[:, ::-1]
                # TODO DANGER np . unique
                arr = np.unique(arr, axis=0)
                # print(f"Flipping axes for {vid} in MTG.")
                # print(f"Coordinates: {arr}")
            paths.append(arr)
        return paths
