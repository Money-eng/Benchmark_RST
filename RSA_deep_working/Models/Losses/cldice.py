# cl dice loss
from .clDice.cldice_loss.pytorch.cldice import soft_cldice


class CLDice(soft_cldice):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def forward(self, inputs, targets):
        return super().forward(inputs, targets)
