import torch
from .dice_loss import DiceLoss
from .bce_dice_loss import BCEDiceLoss

LOSS_FACTORIES = {
    "bce": torch.nn.BCELoss,
    "dice": DiceLoss,
    "bce_dice": BCEDiceLoss
}

def get_loss(loss_config: dict) -> torch.nn.Module:
    """
    Returns the loss function corresponding to loss_config["name"]
    """
    try:
        return LOSS_FACTORIES[loss_config["name"]](**loss_config["params"])
    except KeyError:
        raise ValueError(f"Unknown loss: {loss_config['name']}")