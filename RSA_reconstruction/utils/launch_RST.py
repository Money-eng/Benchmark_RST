import os
import shutil
import subprocess
import tempfile

import numpy as np
import tifffile as tiff
import torch
from rsml import rsml2mtg
from utils.root_System_class import RootSystem


def preprocess_RST_pipeline(
        prediction: torch.Tensor
):
    """
    À partir du tenseur de prédiction (batch de masques), crée un date_map unique,
    copie les autres fichiers nécessaires, et prépare les dossiers d'entrée/sortie pour RST.

    Args:
        prediction (torch.Tensor): Tensor de forme (batch_size, 1, H, W) avec valeurs binaires.

    Returns:
        pred_datemap (np.ndarray): Map des dates (uint8) de forme (H, W).
        input_dir (str): Chemin temporaire vers le dossier d'entrée pour RST.
        output_dir (str): Chemin temporaire vers le dossier de sortie pour RST.
        obs_hours (float): Temps d'observation extrait du RSML ground truth (sera rempli après chargement MTG GT).
    """
    temp_name = "./temps_" + str(os.getpid()) + '/'
    os.makedirs(temp_name, exist_ok=True)
    input_dir = tempfile.mkdtemp(prefix="rst_input_", dir=temp_name)

    prediction_np = prediction.cpu().numpy().astype(np.uint8)
    batch_size, _, height, width = prediction_np.shape

    pred_datemap = np.zeros((height, width), dtype=np.uint8)
    for i in range(batch_size):
        mask_i = (prediction_np[i, 0] > 0) & (pred_datemap == 0)
        pred_datemap[mask_i] = i + 1

    # Sauvegarde date_map au format float32
    input_file = os.path.join(input_dir, "40_date_map.tif")
    tiff.imwrite(input_file, pred_datemap.astype(np.float32))

    # Crée le dossier de sortie
    output_dir = tempfile.mkdtemp(prefix="rst_output_", dir=temp_name)

    # obs_hours est déterminé à partir du RSML ground truth
    # Pour le moment, on retourne None et l'appelant doit le remplir après chargement GT
    return pred_datemap, input_dir, output_dir, None


from pyvirtualdisplay import Display


def generate_graph_with_java(
        input_path: str,
        output_dir: str,
        acq_times: str,
        jar_path: str = "/home/loai/Documents/code/RSMLExtraction/RootSystemTracker/target/rootsystemtracker-1.6.1-jar-with-dependencies.jar",
        expected_filename: str = "61_graph.rsml",
        timeout: int = 120,
):
    """
    Exécute le pipeline Java pour reconstruire un graphe et retourne le chemin du fichier généré.
    Fonction prête à être utilisée en parallèle (multiprocessing, joblib...).

    Args:
        input_path (str): Chemin vers le dossier d'entrée.
        output_dir (str): Dossier de sortie.
        acq_times (list): Temps d’acquisition, séparés par des virgules.
        jar_path (str): Chemin vers le JAR.
        expected_filename (str): Nom du fichier à chercher dans output_dir.
        timeout (int): Temps max d’attente en secondes.

    Returns:
        str | None: Chemin complet du fichier généré, ou None si échec.
    """
    with Display(visible=False, size=(1024, 768)) as _:
        cmd = [
            "java",
            "-cp",
            jar_path,
            "io.github.rocsg.rootsystemtracker.PipelineActionsHandler",
            f"--input={input_path}",
            f"--output={output_dir}",
            f"--acqTimes={acq_times}",
        ]
        try:
            # On ne log pas stdout, juste les vraies erreurs
            _ = subprocess.run(cmd, capture_output=False, text=False, timeout=timeout)
        except Exception as e:
            print(f"[ERREUR] Java failed for {input_path} → {e}")
            return None

    # Vérifie la présence du fichier attendu
    expected_path = os.path.join(output_dir, expected_filename)
    if os.path.exists(expected_path):
        return expected_path
    else:
        print(f"[ERREUR] Fichier attendu non trouvé : {expected_path}")
        return None


def process_date_map(
        mtg_paths: list,
        predictions: torch.Tensor,
        save_path: str,
        jar_path: str = "/home/loai/Documents/code/RSMLExtraction/RootSystemTracker/target/rootsystemtracker-1.6.1-jar-with-dependencies.jar"
):
    """
    Traite un batch issu de series_val_loader (ou équivalent) et renvoie deux MTG :
      - ground truth (chargé directement depuis le fichier RSML indiqué par batch['mtg'])
      - prédiction (issue du pipeline RST sur le masque prédit)

    Args:
        batch (tuple): (images, masks, time, mtg_paths)
            - images (torch.Tensor) : non utilisée ici, mais conservée pour cohérence.
            - masks (torch.Tensor) : tenseur de prédictions, shape (B, 1, H, W)
            - time (Any) : informations temporelles brutes (non utilisées explicitement ici).
            - mtg_paths (list of str) : liste (taille B) de chemins vers les fichiers RSML ground truth.
        base_data_dir (str): Répertoire de base où se trouvent les fichiers associés (images, date_map, etc.)
        jar_path (str): Chemin vers le JAR RST.

    Returns:
        mtg_gt (rsml.MTG): MTG ground truth (premier élément du batch).
        mtg_pred (rsml.MTG): MTG prédit par RST pour le premier élément du batch.
    """
    # On prend le premier élément du batch
    mtg_gt_path = mtg_paths[0]
    if not os.path.exists(mtg_gt_path):
        raise FileNotFoundError(f"Fichier GT RSML introuvable : {mtg_gt_path}")

    mtg_gt = rsml2mtg(mtg_gt_path)
    metadata_gt = mtg_gt.graph_properties().get('metadata', {})
    obs_hours = metadata_gt.get('observation-hours', None)
    if obs_hours is None:
        raise KeyError("Clé 'observation-hours' manquante dans le RSML GT.")

    pred_datemap, input_dir, output_dir, _ = preprocess_RST_pipeline(predictions)
    # Copier tous les fichiers nécessaires depuis le dossier contenant le RSML GT vers input_dir,
    # SAUF date_map (qu'on a déjà générée).
    data_input_dir = os.path.dirname(mtg_gt_path)
    for item in os.listdir(data_input_dir):
        src = os.path.join(data_input_dir, item)
        dst = os.path.join(input_dir, item)
        if item == "40_date_map.tif":
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    if isinstance(obs_hours, (list, tuple, np.ndarray)):
        acq_str = ",".join(str(h) for h in obs_hours)
    else:
        acq_str = str(obs_hours)
    generated_rsml = generate_graph_with_java(
        input_path=input_dir,
        output_dir=output_dir,
        acq_times=acq_str,
        jar_path=jar_path,
        expected_filename="61_graph.rsml",
        timeout=500
    )
    if generated_rsml is None:
        raise RuntimeError(f"Échec génération RSML prédiction pour {input_dir}")

    # 2.4) Charger MTG prédit en passant directement pred_datemap pour éviter rechargement
    rsystem_pred = RootSystem(folder_path=output_dir, date_map=pred_datemap)
    mtg_pred = rsystem_pred.mtg

    # save predicted mtg in the save_path
    os.makedirs(save_path, exist_ok=True)
    rsystem_pred.save2folder(save_path, save_date_map=True)

    # free up resources
    shutil.rmtree(input_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)

    # Retourne les deux MTG
    return mtg_gt, mtg_pred
