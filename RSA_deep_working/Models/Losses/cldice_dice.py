# cl dice loss
from .clDice.cldice_loss.pytorch.cldice import soft_dice_cldice

class CLDice_Dice(soft_dice_cldice):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def forward(self, inputs, targets):
        return super().forward(inputs, targets)