from __future__ import annotations

# --- Standard library imports ------------------------------------------------
import argparse
import copy
import os
from pathlib import Path

# --- Third‑party imports -----------------------------------------------------
import torch
import yaml
from torch.nn import DataParallel

# --- Local package imports ---------------------------------------------------
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
from utils.mask_of_interest import roi_fnc
from utils.misc import SEED, get_device, set_seed

# --------------------------------------------------------------------------- #
#                               GLOBAL SETTINGS                               #
# --------------------------------------------------------------------------- #
set_seed(SEED)
N_GPUS: int = torch.cuda.device_count()
DEFAULT_CFG: Path = Path(__file__).with_name("config.yml")

# Optuna search space
SEARCH_SPACE: dict[str, tuple | list] = {
    "learning_rate": (1e-3, 1e-1),  # logarithmic scale
    "weight_decay": (1e-7, 1e-4),  # logarithmic scale
    "optimizer": ["adamw", "adam"],
}


def load_config(cfg_path: Path | str) -> dict:
    """Load and return the YAML configuration dictionary."""
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict) -> tuple:
    """Build and return (train_loader, val_loader, test_loader)."""
    patch_size: int = cfg["data"]["patch_size"]
    transforms = [
        get_train_img_transform_1(patch_size=patch_size),
        get_train_img_transform_2(patch_size=patch_size),
        get_train_img_transform_3(patch_size=patch_size),
        get__val_test_img_transform(),
    ]
    return create_dataloader(
        base_directory=cfg["data"]["base_dir"],
        img_transforms=transforms,
        batch_size=int(cfg["data"].get("batch_size", 32))
    )


def build_optimizer(
        name: str, params, lr: float, wd: float
) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)


def main() -> None:
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
    cfg_path = Path(args.config) if args.config else DEFAULT_CFG
    cfg = load_config(cfg_path)

    # 1) I/O setup, device, loggers
    device = get_device(preferred=cfg["training"].get("device", "cuda"))
    log_dir = Path(cfg["training"]["log_dir"]) / \
              f"{cfg['model']['name']}_{cfg['loss']['name']}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(
        log_dir / f"{cfg['model']['name']}_{cfg['loss']['name']}.log")
    tb_logger = TensorboardLogger(log_dir / "tensorboard_logs")
    profile_dir = os.path.join(log_dir, "profile")
    os.makedirs(profile_dir, exist_ok=True)

    # 2) DataLoaders
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    metric_logger_path = os.path.join(log_dir, "metrics")
    os.makedirs(metric_logger_path, exist_ok=True)

    # 4) Best hyper‑parameter configuration
    best_cfg = copy.deepcopy(cfg)

    torch.cuda.empty_cache()  # free memory before the real run

    # 5) Final model, loss, optimizer
    model = get_model(best_cfg["model"])
    if N_GPUS > 1:
        model = DataParallel(model)
    model = model.to(device)

    criterion = get_loss(best_cfg["loss"]).to(device)
    optimizer = build_optimizer(
        name=best_cfg["optimizer"]["name"],
        params=model.parameters(),
        lr=float(best_cfg["optimizer"]["learning_rate"]),
        wd=float(best_cfg["optimizer"]["weight_decay"]),
    )

    # 6) Final Evaluator and Trainer
    evaluator = Evaluator(
        model=model,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        criterion=criterion,
        metrics=get_metrics(best_cfg["metrics"]),
        device=device,
        logger=logger,
        threshold=best_cfg["metrics"].get("threshold_4_binarize", 0.5),
        tb_logger=tb_logger,
        patch_size=best_cfg["data"].get("patch_size", 512),
        log_metric_path=metric_logger_path,
        profile_dir=profile_dir,
        roi_fnc=roi_fnc,
        compute_cpu_metrics=False,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=best_cfg,
        evaluator=evaluator,
        logger=logger,
        tb_logger=tb_logger,
        checkpoint_dir=os.path.join(best_cfg["training"].get("checkpoint_dir", os.path.join(log_dir, "checkpoints")),
                                    f"{best_cfg['model']['name']}_{best_cfg['loss']['name']}"),
        device=device,
        epochs=best_cfg["training"].get("epochs", 200),
        epochs_btw_eval=best_cfg["training"].get("epochs_btw_eval", 5),
        do_evaluation=True,
        profile_dir=profile_dir,
    )

    evaluator.evaluate()  # baseline before training
    logger.info("Starting final training…")
    trainer.train()


if __name__ == "__main__":
    main()
