import torch

from .bce_loss import BCEDiceLoss
from .bce_dice_loss import BCEDiceLoss
from .dice_loss import DiceLoss
from .generalized_dice_loss import GeneralizedDiceLoss

LOSS_FACTORIES = {
    "bce": BCEDiceLoss,
    "dice": DiceLoss,
    "bce_dice": BCEDiceLoss,
    "generalized_dice": GeneralizedDiceLoss
}


def get_loss(loss_config: dict) -> torch.nn.Module:
    """
    Returns the loss function corresponding to loss_config["name"]
    """
    try:
        return LOSS_FACTORIES[loss_config["name"].lower()](**loss_config["params"])
    except KeyError:
        raise ValueError(f"Unknown loss: {loss_config['name']}")
