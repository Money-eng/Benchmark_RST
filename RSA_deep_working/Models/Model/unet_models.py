import segmentation_models_pytorch as smp
import torch

def get_unet(in_channels=1, classes=1, encoder="resnet34", weights=None):
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=classes
    )
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))