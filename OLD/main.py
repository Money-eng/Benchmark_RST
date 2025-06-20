from __future__ import annotations

import networkx as nx
from Data_loader import dataloaders
from TDA.filtration import geodesic_filtration, euclidean_filtration
from TDA.persistence_diagram import diagram_from_filtration
from hydroroot.analysis import intercept
from openalea.mtg import MTG
from pathlib import Path
from tqdm import tqdm  # your existing loader package
from util.mtgutils import mtg_at_time_t

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
    )  # 3 er loader = train, val, test sets (shuffled images)
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
        from hydroroot.hydro_io import import_rsml_to_discrete_mtg

        mtg_continuous = rsml2mtg(mtg_paths[-1])
        plant_ids = mtg_continuous.vertices(scale=1)
        print("mtg :", mtg_paths[-1])
        print("plant_ids", plant_ids)
        import matplotlib.pyplot as plt

        plant_id = plant_ids[0]
        sub_mtg_contninous = mtg_continuous.sub_mtg(plant_id)
        intercepto_all = []
        times = [i for i in range(1, 29)]
        fig, ax = plt.subplots()
        for time in times:
            time_mtg0 = mtg_at_time_t(sub_mtg_contninous, time)
            time_mtg = import_rsml_to_discrete_mtg(
                time_mtg0)  # ATTENTION MODIF DANS LE CODE SOURCE DANGER TODO ERROR LIGNE 215
            list_lentghs = [i * 1e-4 for i in range(0, 25000)]
            intercepto = intercept(g=time_mtg, dists=list_lentghs, dl=3e-3, max_order=None)
            intercepto_all.append(intercepto)
            ax.plot(list_lentghs, intercepto, label=f"time {time}")
        ax.set_xlabel("length")
        ax.set_ylabel("intercepto")
        plt.title("intercepto vs length for 1 plant in time")
        plt.legend()
        plt.show()
        import numpy as np

        lengths = np.array([i * 1e-4 for i in range(0, 25000)])
        Z = np.stack(intercepto_all)  # shape (T, L)
        # plot 3D of size 28 * 25000 * max_length 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(intercepto_all)):
            ax.plot(lengths, intercepto_all[i], zs=i, zdir='y', label=f"time {i}")
        ax.set_xlabel("length")
        ax.set_ylabel("time")
        ax.set_zlabel("intercepto")
        plt.title("intercepto vs length for 1 plant in time")
        plt.legend()
        plt.show()


def bifule():
    # suplot of length of the plant
    fig, ax = plt.subplots()
    for plant_id in plant_ids:
        sub_mtg_contninous = mtg_continuous.sub_mtg(plant_id)
        mtg = import_rsml_to_discrete_mtg(sub_mtg_contninous)
        # determine longest path in mtg
        list_lentghs = [i * 1e-4 for i in range(0, 25000)]
        intercepto = intercept(g=mtg, dists=list_lentghs, dl=3e-3, max_order=None)
        # plot the intercepto
        ax.plot(list_lentghs, intercepto, label=f"plant {plant_id}")
    ax.set_xlabel("length")
    ax.set_ylabel("intercepto")
    plt.title("intercepto vs length")
    plt.legend()
    plt.show()

    print("intercepto", intercepto)
