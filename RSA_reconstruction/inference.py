import argparse
from pathlib import Path

import torch 
import yaml
from torch.nn import DataParallel

from DataLoaders.transforms import (
    get__val_test_img_transform,
)
from Models import get_model
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

    name = cfg["model"]["name"] + "_" + cfg["loss"]["name"]
    # load an image
    from tifffile import tifffile
    images = tifffile.imread("/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5/loai/Jean_trap-test/11/22_registered_stack.tif")
    print(images.shape)
    import numpy as np
    
    prediction = torch.zeros((images.shape[0], images.shape[1], images.shape[2]), dtype=torch.float32)
    for i in range(images.shape[0]):
        image = images[i]
        # scale image : size / 2 
        #image = image[::2, ::2]
        print(f"Image {i} shape: {image.shape}, dtype: {image.dtype}")
        image = image.astype("float32")
        transform = get__val_test_img_transform()
        image = transform(image=image)["image"].unsqueeze(0)
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        image = image.to(device)
        pred = _infer(image, model)
        prediction[i] = pred.detach().cpu()
        print(f"Pred shape: {pred.shape}, dtype: {pred.dtype}")
        #pred = torch.sigmoid(pred)
        
        
        print(pred.min(), pred.max(), pred.mean(), pred.std())
        # save heatmap as float32 tiff
        pred = pred.detach().cpu().numpy().astype("float32")
        
        from os import makedirs
        makedirs(f"/home/loai/Documents/code/RSMLExtraction/temp/{name}", exist_ok=True)
        tifffile.imwrite(
            f"/home/loai/Documents/code/RSMLExtraction/temp/{name}/heatmap_{i}.tif",
            pred,
        )
        
    from utils.launch_RST import assemble_date_map
    pred_datemap = assemble_date_map(torch.tensor(prediction).unsqueeze(1))
    # save date_map as uint8 tiff
    pred_datemap = pred_datemap.astype(np.uint8)
    from os import makedirs 
    makedirs(f"/home/loai/Documents/code/RSMLExtraction/temp/{name}", exist_ok=True)
    tifffile.imwrite(
        f"/home/loai/Documents/code/RSMLExtraction/temp/{name}/date_map.tif",
        pred_datemap,
    )

    # save output in

if __name__ == "__main__":
    main()
