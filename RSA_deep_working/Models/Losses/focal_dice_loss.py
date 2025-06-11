from monai.losses import DiceFocalLoss

class BCEDiceLoss(DiceFocalLoss):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def forward(self, inputs, targets):
        return super().forward(inputs, targets)