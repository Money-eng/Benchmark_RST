import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = float(smooth)

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)