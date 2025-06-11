import segmentation_models_pytorch as smp
import torch.nn as nn
from torch import sigmoid


class Segformer(nn.Module):
    """
    Wrapper around segmentation_models_pytorch's Segformer.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output classes/channels.
        encoder_name (str): Encoder backbone (e.g. 'resnet34').
        encoder_weights (str|None): Pretrained weights, e.g. 'imagenet'.
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
        activation = None if return_logits else "sigmoid"
        self.model = smp.Segformer(
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