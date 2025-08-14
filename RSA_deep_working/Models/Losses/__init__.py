import torch
from utils.misc import set_seed, SEED

from .bce_dice_loss import BCEDiceLoss
from .bce_loss import BCE
from .cldice import CLDice
from .cldice_dice import CLDice_Dice
from .dice_loss import DiceLoss
from .generalized_dice_loss import GeneralizedDiceLoss

set_seed(SEED)

LOSS_FACTORIES = {
    "bce": BCE,
    "dice": DiceLoss,
    "bce_dice": BCEDiceLoss,
    "generalized_dice": GeneralizedDiceLoss,
    "cldice": CLDice,
    "cldice_dice": CLDice_Dice
}


def get_loss(loss_config: dict) -> torch.nn.Module:
    """
    Returns the loss function corresponding to loss_config["name"]
    """
    try:
        return LOSS_FACTORIES[loss_config["name"].lower()](**loss_config["params"])
    except KeyError:
        raise ValueError(f"Unknown loss: {loss_config['name']}")
