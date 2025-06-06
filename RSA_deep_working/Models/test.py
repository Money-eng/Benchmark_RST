import os
import yaml
import torch
from DataLoaders.dataloaders import create_dataloader
from torchvision import transforms
from utils.launch_RST import process_batch
from utils.root_System_class import mtg2rsml


if __name__ == "__main__":
    # Chargement de la configuration
    cfg_path = "RSA_deep_working/Models/config.yml"
    assert os.path.exists(cfg_path), f"Le fichier de config n'existe pas : {cfg_path}"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # Préparation des transformations
    pad = transforms.Pad(padding=(0, 0, 28, 18), fill=0)
    img_transform = transforms.Compose([pad, transforms.ToTensor()])
    mask_transform = transforms.Compose([pad, transforms.ToTensor()])

    # Création des DataLoaders
    train_loader, val_loader, test_loader, series_val_loader, series_test_loader = create_dataloader(
        base_directory=config["data"]["base_dir"],
        img_transform=img_transform,
        mask_transform_image=mask_transform,
        mask_transform_series=mask_transform,
        default_batch_size=int(config["data"].get("batch_size", 32)),
    )

    # Récupérer un batch de validation en série
    batch = next(iter(series_val_loader))
    # Traiter le batch pour obtenir MTG GT et MTG prédiction
    mtg_gt, mtg_pred = process_batch(
        batch=batch,
        base_data_dir=config["data"]["base_dir"],
        jar_path=config["rst"]["jar_path"] if config["rst"].get("jar_path") else "/home/loai/Documents/code/RSMLExtraction/RootSystemTracker/target/rootsystemtracker-1.6.1-jar-with-dependencies.jar"
    )

    # Exemple de sauvegarde des deux MTG dans un dossier de sortie
    output_root = config["outputs"].get("results_dir", "./results")
    os.makedirs(output_root, exist_ok=True)

    # Sauvegarde GT
    gt_folder = os.path.join(output_root, "GT_mtg")
    os.makedirs(gt_folder, exist_ok=True)
    mtg2rsml(mtg_gt, os.path.join(gt_folder, "61_graph_gt.rsml"))
    print(f"MTG ground truth sauvegardé dans : {gt_folder}/61_graph_gt.rsml")if __name__ == "__main__":
        # Chargement de la configuration
        cfg_path = "RSA_deep_working/Models/config.yml"
        assert os.path.exists(cfg_path), f"Le fichier de config n'existe pas : {cfg_path}"
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
    
        # Préparation des transformations
        pad = transforms.Pad(padding=(0, 0, 28, 18), fill=0)
        img_transform = transforms.Compose([pad, transforms.ToTensor()])
        mask_transform = transforms.Compose([pad, transforms.ToTensor()])
    
        # Création des DataLoaders
        train_loader, val_loader, test_loader, series_val_loader, series_test_loader = create_dataloader(
            base_directory=config["data"]["base_dir"],
            img_transform=img_transform,
            mask_transform_image=mask_transform,
            mask_transform_series=mask_transform,
            default_batch_size=int(config["data"].get("batch_size", 32)),
        )
    
        # Récupérer un batch de validation en série
        batch = next(iter(series_val_loader))
        # Traiter le batch pour obtenir MTG GT et MTG prédiction
        mtg_gt, mtg_pred = process_batch(
            batch=batch,
            base_data_dir=config["data"]["base_dir"],
            jar_path=config["rst"]["jar_path"] if config["rst"].get("jar_path") else "/home/loai/Documents/code/RSMLExtraction/RootSystemTracker/target/rootsystemtracker-1.6.1-jar-with-dependencies.jar"
        )
    
        # Exemple de sauvegarde des deux MTG dans un dossier de sortie
        output_root = config["outputs"].get("results_dir", "./results")
        os.makedirs(output_root, exist_ok=True)
    
        # Sauvegarde GT
        gt_folder = os.path.join(output_root, "GT_mtg")
        os.makedirs(gt_folder, exist_ok=True)
        mtg2rsml(mtg_gt, os.path.join(gt_folder, "61_graph_gt.rsml"))
        print(f"MTG ground truth sauvegardé dans : {gt_folder}/61_graph_gt.rsml")
    
        # Sauvegarde PRÉDICTION
        pred_folder = os.path.join(output_root, "Pred_mtg")
        os.makedirs(pred_folder, exist_ok=True)
        mtg2rsml(mtg_pred, os.path.join(pred_folder, "61_graph_pred.rsml"))
        print(f"MTG prédiction sauvegardé dans : {pred_folder}/61_graph_pred.rsml")
    
        # Affichage des informations principales
        print("--> MTG Ground Truth:")
        print(f"  - Nombre de sommets : {len(mtg_gt.vertices())}")
        print("  - Times (premiers 5) :", mtg_gt.property('time')[:5] if mtg_gt.property('time') else "N/A")
    
        print("--> MTG Prédiction:")
        print(f"  - Nombre de sommets : {len(mtg_pred.vertices())}")
        print("  - Times (premiers 5) :", mtg_pred.property('time')[:5] if mtg_pred.property('time') else "N/A")

    # Sauvegarde PRÉDICTION
    pred_folder = os.path.join(output_root, "Pred_mtg")
    os.makedirs(pred_folder, exist_ok=True)
    mtg2rsml(mtg_pred, os.path.join(pred_folder, "61_graph_pred.rsml"))
    print(f"MTG prédiction sauvegardé dans : {pred_folder}/61_graph_pred.rsml")

    # Affichage des informations principales
    print("--> MTG Ground Truth:")
    print(f"  - Nombre de sommets : {len(mtg_gt.vertices())}")
    print("  - Times (premiers 5) :", mtg_gt.property('time')[:5] if mtg_gt.property('time') else "N/A")

    print("--> MTG Prédiction:")
    print(f"  - Nombre de sommets : {len(mtg_pred.vertices())}")
    print("  - Times (premiers 5) :", mtg_pred.property('time')[:5] if mtg_pred.property('time') else "N/A")