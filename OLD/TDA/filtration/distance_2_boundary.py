#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filtration.py
-------------
Utilities to convert binary or probabilistic masks into scalar filtrations
for TDA pipelines (e.g., GUDHI, giotto-tda).

Available filtrations:
  - euclidean_filtration: based on Euclidean distance transform.
  - geodesic_filtration: based on geodesic distance from a seed point.
"""
from __future__ import annotations

import logging
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Union

from visu import visualize_filtration

logger = logging.getLogger(__name__)


def euclidean_filtration(
        mask: Union[np.ndarray, list],
        inverse: bool = True,
        visualize: bool = False
) -> np.ndarray:
    """
    Compute a Euclidean-distance based filtration from a binary or probability mask.

    Parameters
    ----------
    mask : array-like of shape (H, W)
        Boolean or numeric mask. Non-zero (or True) values are considered foreground.
    inverse : bool, default=True
        If True, returns max_distance - distance, so that interior pixels
        have lower filtration values (appear earlier in sub-level sets).
    visualize : bool, default=False
        If True, displays the mask and resulting filtration side by side.

    Returns
    -------
    filtration : np.ndarray of float32, shape (H, W)
        Filtration values for each pixel.
    """
    arr = np.asarray(mask, dtype=bool)
    if not arr.any():
        logger.warning("Input mask is empty. Returning zeros.")
        return np.zeros(arr.shape, dtype=np.float32)

    dist = distance_transform_edt(arr)
    filtration = (dist.max() - dist) if inverse else dist
    filtration = filtration.astype(np.float32)

    if visualize:
        visualize_filtration(arr, filtration, title='Euclidean Filtration')

    return filtration
