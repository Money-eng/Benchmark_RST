from torch.nn import BCELoss


class BCE(BCELoss):
    def __init__(self, **kwargs):
        """
        Initialize the BCE-Dice loss with parameters.
        """
        super().__init__(**kwargs)

    def forward(self, inputs, targets):
        return super().forward(inputs, targets)
