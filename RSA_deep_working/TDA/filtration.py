"""filtration.py
-----------------
Utility functions to turn a binary (or probability) mask into a scalar *filtration* that
can be fed to GUDHI/giotto-tda.

Two filtrations are implemented:

1. **Euclidean distance** to the background (default): inner pixels appear first in the
   sub-level set.
2. **Geodesic distance** inside the mask from a user-supplied seed pixel.

Both return a 2-D ``float`` ``np.ndarray`` of the same shape as the mask.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.graph import MCP_Geometric

def euclidean_filtration(binary_mask: np.ndarray, inverse: bool = True, show: bool = False) -> np.ndarray:
    """Return an Euclidean-distance based filtration.

    Parameters
    ----------
    binary_mask : (H, W) ndarray of bool or int
        Non-zero pixels belong to the object.
    inverse : bool, default ``True``
        If *True* (default) returns ``max_dist - dist`` so that *interior* pixels
        appear earlier (lower values) in a sub-level set filtration.  If *False*,
        returns the raw distance transform.
    """
    binary_mask = binary_mask.astype(bool)
    dist = distance_transform_edt(binary_mask)
    filter = dist.max() - dist if inverse else dist
    # show on one hand the mask and on the other the filteration
    if show:
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np

        # Create a figure with two subplots
        _, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Display the binary mask in the first subplot
        axs[0].imshow(binary_mask, cmap='gray')
        axs[0].set_title('Binary Mask')
        axs[0].axis('off')

        # Display the filteration in the second subplot
        axs[1].imshow(filter, cmap='hot')
        axs[1].set_title('Filtration')
        axs[1].axis('off')

        # Show the plot
        plt.show()
    return filter


def geodesic_filtration(binary_mask: np.ndarray, seed: Tuple[int, int], show: bool = False) -> np.ndarray:
    """Return a *geodesic* distance inside the mask from ``seed``.

    Uses Dijkstra’s algorithm on a pixel graph via
    ``skimage.graph.MCP_Geometric`` (fully connected 8-neighbourhood).

    Pixels outside the mask receive the maximal finite distance so that the
    filtration remains finite everywhere.
    """
    assert binary_mask[seed] == 1, "Seed pixel must be non-zero (1)."

    import numpy as np
    cost = np.where(binary_mask, 1.0, np.inf)
    mcp = MCP_Geometric(cost, fully_connected=True)
    # find_costs returns (costs, traceback).  We only need the cost map.
    geo, _ = mcp.find_costs([seed])

    # Replace \infty (background) with the largest *finite* value so that the
    # filtration is defined everywhere.
    finite_max = np.max(geo[np.isfinite(geo)])
    geo[~np.isfinite(geo)] = finite_max
    
    # show on one hand the mask and on the other the filteration
    if show:
        import matplotlib.pyplot as plt
        import numpy as np

        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Display the binary mask in the first subplot
        axs[0].imshow(binary_mask, cmap='gray')
        axs[0].set_title('Binary Mask')
        axs[0].axis('off')

        # Display the filteration in the second subplot
        axs[1].imshow(geo, cmap='hot')
        axs[1].set_title('Filtration')
        axs[1].axis('off')

        # Show the plot
        plt.show()
    return geo
