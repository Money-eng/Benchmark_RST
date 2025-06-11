import segmentation_models_pytorch as smp
import torch.nn as nn


class UNet(nn.Module):
    """
    Wrapper around segmentation_models_pytorch's UNet.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output classes/channels.
        encoder_name (str): Encoder backbone (e.g. 'resnet34').
        encoder_weights (str|None): Pretrained weights, e.g. 'imagenet'.
        return_logits (bool): If True, returns raw logits; otherwise returns probabilities.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_name: str = "resnet34",
        encoder_weights: str = None,
        decoder_attention_type: str = None,
        return_logits: bool = False,
    ):
        super().__init__()
        # If return_logits=True, activation=None gives raw logits.
        activation = None if return_logits else "sigmoid"
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            decoder_attention_type=decoder_attention_type,
            activation=activation,
        )
        self.return_logits = return_logits

    def forward(self, x):
        return self.model(x)
