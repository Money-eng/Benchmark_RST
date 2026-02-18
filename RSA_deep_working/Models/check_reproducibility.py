import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.nn import DataParallel
from tqdm import tqdm

from DataLoaders.dataloaders import create_dataloader
from DataLoaders.transforms import (
    get_train_img_transform_1,
    get_train_img_transform_2,
    get_train_img_transform_3,
    get__val_test_img_transform,
)
from Model import get_model
from utils.misc import SEED, get_device, set_seed

set_seed(SEED)

def load_config(cfg_path: Path | str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def get_val_dataloader(cfg: dict):
    """Récupère uniquement le val_loader via ta fonction existante."""
    patch_size = cfg["data"]["patch_size"]
    transforms = [
        get_train_img_transform_1(patch_size=patch_size),
        get_train_img_transform_2(patch_size=patch_size),
        get_train_img_transform_3(patch_size=patch_size),
        get__val_test_img_transform(),
    ]
    _, val_loader, _ = create_dataloader(
        base_directory=cfg["data"]["base_dir"],
        img_transforms=transforms,
        batch_size=cfg["data"].get("batch_size", 2),
    )
    return val_loader

def run_inference_pass(cfg, weights_path, dataloader, device, run_name="Run 1"):
    """
    Charge un modèle frais, nettoie les poids (DataParallel fix), et fait l'inférence.
    """
    print(f"\n--- Démarrage {run_name} ---")
    
    # 1. Instanciation du modèle (CPU d'abord)
    print(f"[{run_name}] Construction du modèle...")
    model = get_model(cfg["model"])
    
    # 2. Chargement du checkpoint
    print(f"[{run_name}] Chargement des poids depuis {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu") # On charge sur CPU pour manipuler les dictionnaires
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # --- FIX DATAPARALLEL : Nettoyage des clés 'module.' ---
    # On crée un nouveau dictionnaire en retirant le préfixe 'module.' s'il existe
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            name = k[7:] # On retire les 7 premiers caractères ("module.")
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
            
    # 3. Chargement des poids dans le modèle (avant de le passer sur GPU/DataParallel)
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print(f"[{run_name}] Poids chargés avec succès (strict=True).")
    except RuntimeError as e:
        print(f"[{run_name}] ERREUR de chargement : \n{e}")
        raise e

    # 4. Envoi sur le Device (et DataParallel si nécessaire pour l'exécution)
    if torch.cuda.device_count() > 1:
        print(f"[{run_name}] Wrapping en DataParallel pour l'inférence.")
        model = DataParallel(model)
    
    model = model.to(device)

    # 5. Mode Evaluation (CRITIQUE)
    model.eval()

    # 6. Inférence
    all_preds = []
    print(f"[{run_name}] Inférence en cours...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=run_name):
            imgs = batch[0].to(device)
            preds = model(imgs)
            all_preds.append(preds.detach().cpu())
            
    full_output = torch.cat(all_preds, dim=0)
    
    # Nettoyage
    del model
    del checkpoint
    del state_dict
    del new_state_dict
    torch.cuda.empty_cache()
    
    return full_output

def main():
    parser = argparse.ArgumentParser(description="Vérification de la reproductibilité de l'inférence.")
    parser.add_argument("--config", type=str, default="config.yml", help="Chemin vers le config.yml")
    parser.add_argument("--weights", type=str, required=True, help="Chemin vers le fichier .pth")
    parser.add_argument("--max_samples", type=int, default=None, help="Limiter le nombre de samples pour aller vite (optionnel)")
    
    args = parser.parse_args()
    
    # Configuration
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    device = get_device(preferred=cfg["training"].get("device", "cuda"))
    
    print(f"Utilisation du device : {device}")

    # DataLoader (On l'instancie UNE fois pour garantir que l'ordre des données est le même)
    # Note : Assure-toi que ton val_loader a shuffle=False par défaut ou dans la config
    val_loader = get_val_dataloader(cfg)

    # --- PASSE 1 ---
    output_1 = run_inference_pass(cfg, args.weights, val_loader, device, run_name="Run A")

    # --- PASSE 2 ---
    output_2 = run_inference_pass(cfg, args.weights, val_loader, device, run_name="Run B")

    # --- COMPARAISON ---
    print("\n--- Résultats de la comparaison ---")
    
    # Vérification des dimensions
    if output_1.shape != output_2.shape:
        print(f"ERREUR FATALE : Les dimensions diffèrent ! {output_1.shape} vs {output_2.shape}")
        return

    # Calcul de la différence
    diff = torch.abs(output_1 - output_2)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Différence Max Absolue : {max_diff}")
    print(f"Différence Moyenne     : {mean_diff}")

    # Test strict (tolerance 0 ou epsilon très faible pour le floating point error)
    # 1e-6 est généralement sûr pour float32, 0.0 est possible si tout est déterministe pur
    if max_diff == 0.0:
        print("\n✅ SUCCÈS : Les sorties sont STRICTEMENT identiques (bit-exact).")
    elif max_diff < 1e-7:
        print("\n✅ SUCCÈS : Les sorties sont identiques (aux erreurs d'arrondi FP32 près).")
    else:
        print("\n❌ ÉCHEC : Il y a des différences significatives entre les deux runs.")
        print("Pistes à vérifier :")
        print("1. model.eval() a-t-il été appelé ?")
        print("2. Le DataLoader mélange-t-il les données (shuffle=True) ?")
        print("3. Y a-t-il des opérations non-déterministes dans le modèle (Upsample bi-linear, etc.) ?")
        print("4. Les transforms de validation comportent-elles de l'aléatoire ?")

if __name__ == "__main__":
    main()