import argparse
from pathlib import Path

import torch
import yaml
from torch.nn import DataParallel

from DataLoaders.dataloaders import create_dataloader
from DataLoaders.transforms import (
    get_train_img_transform_1,
    get_train_img_transform_2,
    get_train_img_transform_3,
    get__val_test_img_transform,
)
from Models import get_model
from reconstructor import Reconstructor
from monai.inferers import SlidingWindowInfererAdapt
from utils.misc import SEED, set_seed, get_device

set_seed(SEED)
DEFAULT_CFG: Path = Path(__file__).with_name("config.yml")
DEFAULT_MODEL_PATH = ""


def load_config(cfg_path: Path | str) -> dict:
    """Load and return the YAML configuration dictionary."""
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def _infer(imgs: torch.Tensor, model) -> torch.Tensor:
    """Forward pass with optional sliding-window inference."""
    sw_inferer = SlidingWindowInfererAdapt(
        roi_size=(int(512), int(512)),
        sw_batch_size=4,
        overlap=0.25,
        mode="constant",
    )
    if sw_inferer is None:
        return model(imgs)
    return sw_inferer(inputs=imgs, network=model)


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

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Path to the model checkpoint file. "
            "If omitted, 'model.pth' is searched in the same directory as this script."
        ),
    )

    args = parser.parse_args()
    cfg_path = Path(args.config) if args.config else DEFAULT_CFG
    cfg = load_config(cfg_path)

    model_checkpoints_path = Path(
        args.model_path) if args.model_path else DEFAULT_MODEL_PATH

    device = get_device()
    model = get_model(cfg["model"])
    model = DataParallel(model)
    state_dict = torch.load(
        model_checkpoints_path,
        map_location=device
    )
    model.load_state_dict(state_dict)
    model = model.to(device)

    # load an image
    from tifffile import tifffile
    image = tifffile.imread(
        # shape (T, H, W)
        "/home/loai/Documents/code/RSMLExtraction/temp/22_registered_stack.tif")
    # to pytorch tensor of shape (1, T, H, W)
    image = torch.from_numpy(image[0]).unsqueeze(0).unsqueeze(0).float()
    image = image.to(device)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    pred = _infer(image, model)
    
    # save output in

if __name__ == "__main__":
    main()
