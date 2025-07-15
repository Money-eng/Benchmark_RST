import os
import yaml
import copy
import torch
import optuna
from torch.nn import DataParallel

from DataLoaders.dataloaders import create_dataloader
from DataLoaders.transforms import (
    get_train_img_transform_1,
    get_train_img_transform_2,
    get_train_img_transform_3,
    get__val_test_img_transform,
)
from Losses import get_loss
from Metrics import get_metrics
from Model import get_model
from Training.evaluator import Evaluator
from Training.trainer import Trainer
from utils.logger import get_logger, TensorboardLogger
from utils.misc import get_device, set_seed, SEED

set_seed(SEED)

# 1) Configuration file ------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
cfg_path = os.path.join(current_dir, "config.yml")
assert os.path.exists(
    cfg_path), f"Le fichier de config n'existe pas : {cfg_path}"
with open(cfg_path, "r") as f:
    CONFIG_BASE = yaml.safe_load(f)

# DataLoaders --------------------------------------------------------
train_loader, val_loader, test_loader = create_dataloader(
    base_directory=CONFIG_BASE["data"]["base_dir"],
    img_transforms=[
        get_train_img_transform_1(
            patch_size=CONFIG_BASE["data"]["patch_size"]),
        get_train_img_transform_2(
            patch_size=CONFIG_BASE["data"]["patch_size"]),
        get_train_img_transform_3(
            patch_size=CONFIG_BASE["data"]["patch_size"]),
        get__val_test_img_transform(),
    ],
    batch_size=int(CONFIG_BASE["data"].get("batch_size", 32)),
    generator=torch.Generator().manual_seed(SEED),
)

# 3) Devices & loggers ----------------------------------------------------------
N_GPUS = torch.cuda.device_count()
DEVICE = get_device(preferred=CONFIG_BASE["training"].get("device", "cuda"))
LOG_DIR = os.path.join(CONFIG_BASE["training"]["log_dir"], "optuna")
os.makedirs(LOG_DIR, exist_ok=True)
LOGGER = get_logger(os.path.join(LOG_DIR, "optuna.log"))
TB_LOGGER = TensorboardLogger(os.path.join(LOG_DIR, "tensorboard_logs"))

# 4) Hyper‑parameters -----------------------------------------------------------
SEARCH_SPACE = {
    "learning_rate": (1e-6, 1e-1),  # log scale
    "weight_decay": (1e-8, 1e-1),  # log scale
    "optimizer":     ["adamw", "adam", "sgd"],
}

# 5) Number of epochs for search and final training --------------------------
EPOCHS_SEARCH = CONFIG_BASE["training"].get("optuna_epochs", 10)

# --------------------------------------------------------------------------------------
#  Objective Optuna --------------------------------------------------------------------
# --------------------------------------------------------------------------------------


def objective(trial: optuna.Trial) -> float:
    """Un essai Optuna: renvoie la loss de validation (à minimiser)."""
    config = copy.deepcopy(CONFIG_BASE)

    # 1. Échantillonnage des hyper‑params  ------------------------------------------------
    lr = trial.suggest_float(
        "learning_rate", *SEARCH_SPACE["learning_rate"], log=True)
    wd = trial.suggest_float(
        "weight_decay",  *SEARCH_SPACE["weight_decay"],  log=True)
    opt_name = trial.suggest_categorical(
        "optimizer", SEARCH_SPACE["optimizer"])

    config["optimizer"]["learning_rate"] = lr
    config["optimizer"]["weight_decay"] = wd
    config["optimizer"]["name"] = opt_name

    # 2. Création du modèle / loss / optimiseur -----------------------------------------
    model = get_model(config["model"])
    if N_GPUS > 1:
        model = DataParallel(model)
    model = model.to(DEVICE)

    criterion = get_loss(config["loss"]).to(DEVICE)

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=wd)
    else:  # SGD
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    # 3. Initialisation des métriques / evaluator (utilise val_loader)
    metrics_dict = get_metrics(config["metrics"])
    evaluator = Evaluator(
        model=model,
        val_dataloader=val_loader,
        criterion=criterion,
        test_dataloader=None,
        metrics=metrics_dict,
        device=DEVICE,
        logger=LOGGER,
        threshold=config["metrics"].get("threshold_4_binarize", 0.5),
        tb_logger=None,
        patch_size=config["data"].get("patch_size", 512),
        log_metric_path=None,
    )

    # 4. Entraînement rapide -------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        evaluator=evaluator,
        logger=LOGGER,
        tb_logger=None,
        checkpoint_dir=None,
        device=DEVICE,
        epochs=EPOCHS_SEARCH,
        epochs_btw_eval=50,
        do_evaluation=False
    )

    trainer.train()

    # 5. Évaluation et retour de la métrique (on minimise) -------------------------------
    eval_results = evaluator.evaluate()
    val_loss = eval_results.get(f"val_loss_{criterion.__class__.__name__}", float("inf"))

    # Libère VRAM entre les essais ------------------------------------------------------
    del model, optimizer, criterion, trainer
    torch.cuda.empty_cache()

    return val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective, n_trials=CONFIG_BASE["training"].get("optuna_trials", 30))

    print("\n=== BEST TRIAL ===")
    print("  value (val_loss)", study.best_value)
    print("  params", study.best_params)

    # Sauvegarde dans un fichier pour reproductibilité ----------------------------------
    study_path = os.path.join(LOG_DIR, "study.pkl")
    optuna.study.study_pickle.dump(study, study_path)
    print(f"Étude Optuna sauvegardée dans {study_path}\n")
