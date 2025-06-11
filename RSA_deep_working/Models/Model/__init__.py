import torch

from .segformer import Segformer
from .unet import UNet

MODEL_FACTORIES = {
    "unet": UNet,
    "segformer": Segformer,
}


def get_model(model_config: dict) -> torch.nn.Module:
    name = model_config["name"]
    params = model_config.get("params", {})
    if name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {name}. Known: {list(MODEL_FACTORIES)}")
    try:
        return MODEL_FACTORIES[name](**params)
    except TypeError as e:
        raise TypeError(f"Error instantiating {name} with params {params}: {e}")
