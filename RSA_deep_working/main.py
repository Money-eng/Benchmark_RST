
from __future__ import annotations

from pathlib import Path
from RSA_deep_working.Data_loader import dataloaders
import networkx as nx
from tqdm import tqdm 
from RSA_deep_working.Data_loader import dataloaders  # your existing loader package
from TDA.filtration import geodesic_filtration, euclidean_filtration
from TDA.persistence_diagram import diagram_from_filtration
from openalea.mtg import MTG

OUTPUT_DIR = Path("/home/loai/Documents/code/RSMLExtraction/RSA_deep_working/TDA/output")

if __name__ == "__main__":
    base_directory = "/home/loai/Images/DataTest/UC1_data"
    img_transform = None
    mask_transform_image = None
    mask_transform_series = None

    # Create loaders
    train_loader, val_loader, test_loader, val_loader_series, test_loader_series = dataloaders.create_dataloader(
        base_directory,
        img_transform=img_transform,
        mask_transform_image=mask_transform_image,
        mask_transform_series=mask_transform_series,
        num_workers=8
    ) # 3 er loader = train, val, test sets (shuffled images)
     # 2 der loader = val, test sets (series) (no "batch size = 29", -> intead batch sampler) -> load whole serie that is in the val/test set
    
    # for testing, we will use the val_loader_series
    for batch_idx, batch in enumerate(tqdm(test_loader_series)):
        imgs, masks, times, mtg_paths = batch  # masks: (T, 1, H, W) or (T, H, W)

        # Remove channel dim if present (depends on your transforms)
        if masks.dim() == 4:
            masks = masks.squeeze(1)
        bin_mask = masks[-1].numpy() > 0.5  # Convert to binary mask
        
        # load mtg 
        from rsml import rsml2mtg
        mtg = rsml2mtg(mtg_paths[-1])
        
        graphs = mtg_to_networkx_forest(mtg=mtg)
        