# cl dice loss
from monai.losses import SoftDiceclDiceLoss


class CLDice_Dice(SoftDiceclDiceLoss):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def forward(self, inputs, targets):
        return super().forward(inputs, targets)
