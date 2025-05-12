
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
from Data_loader import Dataloaders, Dataset, Tiff_reader
import numpy as np
import torch
from tqdm import tqdm 
from Data_loader import Dataloaders  # your existing loader package
from TDA.filtration import geodesic_filtration, euclidean_filtration
from TDA.persistence_diagram import diagram_from_filtration

OUTPUT_DIR = Path("/home/loai/Documents/code/RSMLExtraction/RSA_deep_working/TDA/output")

if __name__ == "__main__":
    base_directory = "/home/loai/Images/DataTest/UC1_data"
    img_transform = None
    mask_transform_image = None
    mask_transform_series = None

    # Create loaders
    train_loader, val_loader, test_loader, val_loader_series, test_loader_series = Dataloaders.create_dataloader(
        base_directory,
        img_transform=img_transform,
        mask_transform_image=mask_transform_image,
        mask_transform_series=mask_transform_series,
        num_workers=8
    ) # 3 er loader = train, val, test sets (shuffled images)
     # 2 der loader = val, test sets (series) (no "batch size = 29", -> intead batch sampler) -> load whole serie that is in the val/test set
    
    # for testing, we will use the val_loader_series
    for batch_idx, batch in enumerate(tqdm(val_loader_series)):
        imgs, masks, times, mtg_paths = batch  # masks: (T, 1, H, W) or (T, H, W)

        # Remove channel dim if present (depends on your transforms)
        if masks.dim() == 4:
            masks = masks.squeeze(1)
        bin_mask = masks[-1].numpy() > 0.5  # Convert to binary mask
        
        # load mtg 
        from rsml import rsml2mtg
        mtg = rsml2mtg(mtg_paths[-1])
        geom = mtg.property('geometry') 
        positions = geom[2]
        
        # get all the points in that are in the mask
        points = []
        for i in range(len(positions)):
            # position to int
            positions[i] = (int(positions[i][0]), int(positions[i][1]))
            if bin_mask[positions[i][1], positions[i][0]] == 1:
                points.append((positions[i][1], positions[i][0]))
        points = np.array(points)
        seed = tuple(points[0])  # use the first point as seed
        print(f"Seed point: {seed}")
        
        filtration = geodesic_filtration(bin_mask, seed, show=True)
        # euclidean_filtration(bin_mask, show=True)  # distance-based filtration
        diagrams: Dict[int, np.ndarray] = diagram_from_filtration(filtration)

        # Derive a "series id" from the MTG filename (or fallback to index)
        if isinstance(mtg_paths[0], str):
            series_id = Path(mtg_paths[0]).stem
        else:  # mtg is a tensor (0) when image_with_mtg=False
            series_id = f"series_{batch_idx:03d}"

        # Save diagrams -------------------------------------------------------
        for dim, intervals in diagrams.items():
            np.save(OUTPUT_DIR / f"{series_id}_dim{dim}.npy", intervals)
        
        # show the diagrams
        for dim, intervals in diagrams.items():
            print(f"Diagram for {series_id} (dim {dim}):")
            print(intervals)
            print(f"Number of intervals: {len(intervals)}")
            print(f"Max interval: {np.max(intervals[:, 1] - intervals[:, 0])}")
            print(f"Min interval: {np.min(intervals[:, 1] - intervals[:, 0])}")
            
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        for dim, intervals in diagrams.items():
            plt.scatter(intervals[:, 0], intervals[:, 1], label=f"dim {dim}")
        plt.title(f"Persistence Diagram for {series_id}")
        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.legend()
        plt.grid()
        plt.savefig(OUTPUT_DIR / f"{series_id}_diagram.png")
        plt.show()


    print(f"Done. Diagrams written to {OUTPUT_DIR.resolve()}")

        
        