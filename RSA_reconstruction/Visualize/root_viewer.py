from __future__ import annotations
import numpy as np
import tifffile  # pip install tifffile
import napari  # pip install napari[all]
from magicgui import magicgui  # pip install magicgui
from rsml import rsml2mtg
from pathlib import Path
from openalea.mtg import MTG
from typing import Union, List, Dict, Optional, Sequence


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
        colors: Optional[Dict[str, str]] = None,
        flip_axes: bool = True,
        with_time: bool = True,
    ) -> None:

        self.mtg_dict: Dict[str, Union[MTG]] = {
            "prediction": self._to_mtg(prediction),
            "gt_before_expertized": self._to_mtg(gt_before),
            "gt_expertized": self._to_mtg(gt_expertized),
        }

        self.background = Path(background).expanduser() if background else None
        self.colors = {**self._DEFAULT_COLORS, **(colors or {})}
        self.flip_axes = flip_axes

        self._temporal_shapes: Dict[str, List[List[np.ndarray]]] = {}
        self.with_time = with_time

    def show(self) -> napari.Viewer:
        viewer = napari.Viewer(title="Comparing Root Systems")
        shapes_layers: Dict[str, napari.layers.Shapes] = {}
        n_frames = 1

        if self.background and self.background.exists():
            stack = tifffile.imread(str(self.background))
            viewer.add_image(stack, name="background", blending="additive")
            n_frames = stack.shape[0] if stack.ndim == 3 else 1

        if self.with_time:
            for key, mtg in self.mtg_dict.items():
                for time in range(n_frames):
                    self._temporal_shapes.setdefault(key, []).append(
                        self._mtg_to_paths(mtg, time))

        for key, value in self.mtg_dict.items():
            if self.with_time:
                data0 = self._temporal_shapes[key][0]  # as a start
            else:
                data0 = self._mtg_to_paths(value)

            if not data0:
                print(f"No data found for {key}, skipping visualization.")
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

        @viewer.dims.events.current_step.connect
        def update_layers(event):
            current_step, _, _ = event.value
            for key, layer in shapes_layers.items():
                if self.with_time:
                    layer.data = self._temporal_shapes[key][current_step]
                else:
                    layer.data = self._mtg_to_paths(self.mtg_dict[key])

        # Scale bar & axes
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
        """Constructeur de commodité pour rester rétro‑compatible."""
        return cls(exp_path, before_path, pred_path, **kwargs)

    def _to_mtg(self, obj: Union[str, Path, MTG]) -> Union[MTG, List[MTG]]:
        if hasattr(obj, "property"):
            return obj
        return rsml2mtg(str(obj))

    def _mtg_to_paths(self, mtg: MTG, time: int = -1) -> List[np.ndarray]:
        """Convertit les arêtes *geometry* d’un MTG en liste de chemins numpy."""
        polys = mtg.property("geometry")
        paths: List[np.ndarray] = []
        for key, coords in polys.items():
            arr = np.array(coords)
            arr = arr[:, ::-1]  # (x,y) → (row,col) for napari
            if self.flip_axes:
                if (time < 0):
                    arr = np.unique(arr, axis=0)
                else:
                    time_poly = np.array(mtg.property("time").get(
                        key, None))  # assuming time is sorted
                    arr = arr[time_poly <= time+1]
                    arr = np.unique(arr, axis=0)
                    if (arr.shape[0] == 0):
                        continue
                    elif (arr.shape[0] == 1):
                        # duplicate the last point + dx
                        dx = 1  # or some other value
                        arr = np.concatenate([arr, arr[-1:]])  # + [dx, 0]])
            paths.append(arr)
        return paths
