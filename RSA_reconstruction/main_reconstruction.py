
import argparse
from DataLoaders.dataloaders import create_dataloader
from DataLoaders.transforms import (
    get_train_img_transform_1,
    get_train_img_transform_2,
    get_train_img_transform_3,
    get__val_test_img_transform,
)
import torch
from reconstruction import Reconstructor
from utils.misc import SEED, set_seed, get_device
from pathlib import Path
import yaml
from torch.nn import DataParallel
from Models import get_model

set_seed(SEED)
DEFAULT_CFG: Path = Path(__file__).with_name("config.yml")


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


def load_config(cfg_path: Path | str) -> dict:
    """Load and return the YAML configuration dictionary."""
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


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

    # Build dataloaders
    _, val_loader, test_loader = build_dataloaders(cfg)

    # Model checkpoints folder path
    model_checkpoints_path = cfg.get("model_checkpoints", {}).get(
        "folder_pretrained_path", "Models/Unet_bce")
    # Contains :
    # Model checkpoint that maximized a score over a certain metric
    # Model checkpoint folder ('by_epoch') which saved every epoch
    # Per default, we will take the model of the last epoch

    device = get_device()
    model = get_model(cfg["model"])
    model_checkpoints_name = cfg.get(
        "model_checkpoints", {}).get("name", "Model_X")
    model = DataParallel(model)
    state_dict = torch.load(
        "/home/loai/Documents/code/RSMLExtraction/Results/Checkpoints/Unet_bce/by_epochs/DataParallel_epoch133.pth", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    reconstructor = Reconstructor(
        model=model,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        device=device,
        model_name=model_checkpoints_name,
        threshold=cfg.get("threshold_4_binarize", 0.5),
        patch_size=cfg.get("data", {}).get("patch_size", 512),
        jar_path=cfg.get("rst", {}).get("jar_path", None),
        save_path=cfg.get("data", {}).get("rsml_save_path",
                                          "RSA_reconstruction/Logs/Prediction")
    )

    preds = reconstructor.reconstruct_all()

    print(preds)

if __name__ == "__main__":
    main()
