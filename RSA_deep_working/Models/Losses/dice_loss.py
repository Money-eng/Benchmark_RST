from monai.losses import DiceLoss as MonaiDiceLoss


class DiceLoss(MonaiDiceLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
