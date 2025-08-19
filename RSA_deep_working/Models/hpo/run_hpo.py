from __future__ import annotations

# setup work space to parent directory
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# -------- Standard --------
from pathlib import Path
from typing import Dict, Tuple
import yaml

# -------- 3rd party -------
import torch
from torch.nn import DataParallel

# -------- Projet ---------
from DataLoaders.dataloaders import create_dataloader
from DataLoaders.transforms import (
    get_train_img_transform_1,
    get_train_img_transform_2,
    get_train_img_transform_3,
    get__val_test_img_transform,
)
from Losses import get_loss
from Metrics import get_metric
from Model import get_model
from utils.logger import get_logger
from utils.misc import SEED, get_device, set_seed

# -------- Nouveau ---------
from hpo_search import HPOSearcher

# -----------------------------------------------------------------------------
# 1) CONFIGURATION EN DUR (modifie simplement ces variables)
# -----------------------------------------------------------------------------
# Chemin vers ta config YAML projet
CONFIG_PATH: Path = Path("RSA_deep_working/Models/configs/unet_dice.yml")

# Budget Optuna
N_TRIALS: int = 120  # nombre d'essais
EPOCHS_PER_TRIAL: int = 15  # époques par essai (entraînement court)
EVAL_EVERY: int = 1  # fréquence d'évaluation/pruning (1 = chaque époque)

# Espace HPO (bornes log-uniformes pour LR & WD)
LR_BOUNDS: Tuple[float, float] = (1e-4, 1e-1)
WD_BOUNDS: Tuple[float, float] = (1e-7, 1e-4)
OPTIMIZERS: Tuple[str, ...] = ("adamw", "adam")

# Persister l'étude sur disque ? (sinon en mémoire)
PERSIST_STUDY: bool = True

# Nom/dossiers de sortie
RUN_NAME: str = "HPO_ONLY"  # utilisé pour les dossiers de log

# -----------------------------------------------------------------------------
# 2) Petites fonctions utilitaires (simples)
# -----------------------------------------------------------------------------
set_seed(SEED)
N_GPUS: int = torch.cuda.device_count()


def load_config(cfg_path: Path | str) -> dict:
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict) -> tuple:
    """Construit (train, val, test) comme dans votre projet."""
    patch_size: int = int(cfg["data"]["patch_size"])
    transforms = [
        get_train_img_transform_1(patch_size=patch_size),
        get_train_img_transform_2(patch_size=patch_size),
        get_train_img_transform_3(patch_size=patch_size),
        get__val_test_img_transform(),
    ]
    return create_dataloader(
        base_directory=cfg["data"]["base_dir"],
        img_transforms=transforms,
        batch_size=int(cfg["data"].get("batch_size", 32)),
    )


def make_build_model(cfg: dict):
    """Fabriqueur de modèle pour HPOSearcher (respecte le multi-GPU)."""

    def _build_model(_unused: Dict) -> torch.nn.Module:
        model = get_model(cfg["model"])
        if N_GPUS > 1:
            model = DataParallel(model)
        return model

    return _build_model


def make_build_criterion(cfg: dict):
    def _build_criterion() -> torch.nn.Module:
        return get_loss(cfg["loss"])  # .to(device) géré par SimpleTrainer

    return _build_criterion


def make_build_metric(cfg: dict):
    def _build_metric() -> torch.nn.Module:
        return get_metric(cfg["metrics"]['gpu'][3])

    return _build_metric


# -----------------------------------------------------------------------------
# 3) Programme principal (aucun argparse : tout vient des constantes ci‑dessus)
# -----------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Train/Test a model for root system segmentation in 2D grayscale images."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the YAML configuration file. "
            "If omitted, 'config.yml' is searched in the same directory as this script."
        ),
    )
    args = parser.parse_args()
    cfg_path = Path(args.config) if args.config else CONFIG_PATH
    cfg = load_config(cfg_path)

    device = get_device(preferred=cfg["training"].get("device", "cuda"))
    run_dir = Path(cfg["training"]["log_dir"]) / f"{RUN_NAME}_{cfg['model']['name']}_{cfg['loss']['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(run_dir / "hpo.log")

    # -------- Données --------
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # -------- HPOSearcher --------
    if PERSIST_STUDY:
        study_db = run_dir / "study.db"
        storage = f"sqlite:///{study_db}"
        study_name = f"HPO_{cfg['model']['name']}_{cfg['loss']['name']}"
    else:
        storage = None
        study_name = None

    searcher = HPOSearcher(
        build_model=make_build_model(cfg),
        build_criterion=make_build_criterion(cfg),
        build_metric=make_build_metric(cfg),
        seed=SEED,
        study_storage=storage,
        study_name=study_name,
    )

    logger.info(
        "[HPO] trials=%d | epochs/trial=%d | eval_every=%d | lr=[%.1e, %.1e] | wd=[%.1e, %.1e] | optims=%s",
        N_TRIALS, EPOCHS_PER_TRIAL, EVAL_EVERY, LR_BOUNDS[0], LR_BOUNDS[1], WD_BOUNDS[0], WD_BOUNDS[1],
        ",".join(OPTIMIZERS)
    )

    best = searcher.search(
        train_loader=train_loader,
        val_loader=val_loader,
        n_trials=N_TRIALS,
        epochs_per_trial=EPOCHS_PER_TRIAL,
        eval_every=EVAL_EVERY,
        lr_bounds=LR_BOUNDS,
        wd_bounds=WD_BOUNDS,
        optimizers=OPTIMIZERS,
        device=device,
    )

    # -------- Résultats --------
    logger.info("\n=== BEST TRIAL (HPO only) ===")
    logger.info("  value (val_loss): %s", best.value)
    logger.info("  params: %s", best.params)

    # Sauvegarde YAML pour réutiliser les meilleurs hyperparamètres
    out_yaml = run_dir / "hpo_best.yaml"
    with out_yaml.open("w") as f:
        yaml.safe_dump({"best_val_loss": float(best.value), "best_params": best.params}, f)
    print(f"[HPO] Best params saved to: {out_yaml}")
    if PERSIST_STUDY:
        print(f"[HPO] Study persisted at: {study_db}")


if __name__ == "__main__":
    main()
