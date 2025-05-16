import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union

def visualize_filtration(
    mask: np.ndarray,
    filtration: np.ndarray,
    title: str = 'Filtration'
) -> None:
    """
    Affiche le masque et la filtration côte à côte pour inspection.

    Parameters
    ----------
    mask : np.ndarray of bool
        Masque binaire.
    filtration : np.ndarray of float
        Carte de filtration.
    title : str
        Titre du graphique de filtration.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(mask, cmap='gray', interpolation='none')
    axes[0].set_title('Mask')
    axes[0].axis('off')

    im = axes[1].imshow(filtration, cmap='hot', interpolation='none')
    axes[1].set_title(title)
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()