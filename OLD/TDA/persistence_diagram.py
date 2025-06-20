"""persistence_diagram.py
--------------------------------
Wrappers around **GUDHI** to transform a 2D filtration into persistence
intervals (a.k.a. *barcode* / *diagram*) and to compute distances between
diagrams.

Only cubical complexes are used – perfect for regular image grids.
"""
from __future__ import annotations

import gudhi as gd
import numpy as np
from typing import Iterable, Dict

__all__ = [
    "diagram_from_filtration",
    "bottleneck_distance",
]


def diagram_from_filtration(
        filtration: np.ndarray,
        *,
        coeff_field: int = 2,
        dims: Iterable[int] = (0, 1),
) -> Dict[int, np.ndarray]:
    """Return persistence intervals for selected *homology dimensions*.

    Parameters
    ----------
    filtration : (H, W) ndarray (float)
        Scalar values defining the *sub-level set* filtration.
    coeff_field : int, optional
        Field for homology computations (prime).  Default is 2.
    dims : iterable of int, optional
        Homology dimensions to return.  Default ``(0, 1)``.
    """
    # GUDHI expects the Y‑axis inverted compared to image convention
    cc = gd.CubicalComplex(top_dimensional_cells=filtration[::-1, :])
    cc.compute_persistence(homology_coeff_field=coeff_field)
    return {d: np.asarray(cc.persistence_intervals_in_dimension(d)) for d in dims}


def bottleneck_distance(diag1: np.ndarray, diag2: np.ndarray, epsilon: float = 0.0) -> float:
    """Compute the *bottleneck distance* between two diagrams (same dim).

    ``epsilon`` lets you approximate the true distance faster when tolerated.
    """
    return gd.bottleneck_distance(diag1, diag2, epsilon)
