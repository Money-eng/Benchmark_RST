import torch.nn as nn
from .dice_loss import DiceLoss


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=0.5, smooth=1.0, **kwargs):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=smooth) # DiceLoss class from dice_loss.py
        self.weight = weight

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.weight * bce_loss + (1 - self.weight) * dice_loss