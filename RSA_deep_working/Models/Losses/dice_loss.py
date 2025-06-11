from monai.losses import DiceLoss

class DiceLoss(DiceLoss):
    def __init__(self, **kwargs):
        """
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
        """
        super().__init__(kwargs)

    def forward(self, inputs, targets):
        return super().forward(inputs, targets)



"""OLD
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = float(smooth)

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
"""