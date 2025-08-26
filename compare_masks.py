#!/usr/bin/env python3
"""
Script de comparaison des ratios de pixels différents entre deux modèles (Unet vs Segformer).
Calcule pour chaque échantillon du Test et du Val :
 - ratio de pixels différents (différentes intensités) par rapport au total des pixels
 - ratio de pixels différents dont l’intensité dans Unet est non nulle par rapport au nombre de pixels non nuls dans Unet
Puis affiche la moyenne de ces ratios pour chaque set (Test et Val).
"""
import argparse
from pathlib import Path
import numpy as np
import tifffile

def compare_images(path1: Path, path2: Path):
    """
    Compare deux images TIFF et renvoie :
    - diff_ratio            : diff_count / total_pixels
    - nonzero_diff_ratio    : nonzero_diff_count / total_nonzero_pixels_in_img1
    """
    # Lecture des images TIFF
    img1 = tifffile.imread(path1)
    img2 = tifffile.imread(path2)

    # Vérification des dimensions
    if img1.shape != img2.shape:
        raise ValueError(
            f"Dimensions différentes : {path1} ({img1.shape}) vs {path2} ({img2.shape})"
        )

    total_pixels = img1.size
    total_nonzero_base = np.count_nonzero(img1)

    # Masque des pixels différents
    diff_mask = img1 != img2
    diff_count = np.count_nonzero(diff_mask)

    # Masque des pixels différents et base non nulle
    nonzero_diff_count = np.count_nonzero(diff_mask & (img1 != 0))

    # Ratios
    diff_ratio = diff_count / total_pixels
    nonzero_diff_ratio = nonzero_diff_count / total_nonzero_base if total_nonzero_base > 0 else 0.0

    return diff_ratio, nonzero_diff_ratio


def process_set(model1_root: Path, model2_root: Path, set_name: str):
    print(f"\n=== Set: {set_name} ===")
    dir1 = model1_root / set_name
    dir2 = model2_root / set_name

    samples = sorted([
        d.name for d in dir1.iterdir()
        if d.is_dir() and (dir2 / d.name).is_dir()
    ])
    ratios = []
    nonzero_ratios = []

    for sample in samples:
        f1 = dir1 / sample / '40_date_map.tif'
        f2 = dir2 / sample / '40_date_map.tif'
        if not f1.exists() or not f2.exists():
            print(f"⚠️ Fichier manquant pour {sample} dans {set_name}")
            continue

        diff_ratio, nonzero_ratio = compare_images(f1, f2)
        ratios.append(diff_ratio)
        nonzero_ratios.append(nonzero_ratio)
        print(
            f"{sample}: "
            f"diff_ratio={diff_ratio:.6f}, "
            f"nonzero_diff_ratio={nonzero_ratio:.6f}"
        )

    if ratios:
        avg_ratio = np.mean(ratios)
        avg_nonzero_ratio = np.mean(nonzero_ratios)
        print(
            f"\nMoyenne sur {len(ratios)} échantillons: "
            f"diff_ratio={100 * avg_ratio:.6f} %, "
            f"nonzero_diff_ratio={100 * avg_nonzero_ratio:.6f} %"
        )
    else:
        print("Aucun échantillon traité.")


if __name__ == '__main__':
    model1 = Path('/home/loai/Documents/code/RSMLExtraction/Results/Reconstruction_0.55/Unet_cldice_dice')
    model2 = Path('/home/loai/Documents/code/RSMLExtraction/Results/Reconstruction_0.55/Segformer_bce')

    # Execution pour les deux sets
    for set_name in ['Test', 'Val']:
        process_set(model1, model2, set_name)
