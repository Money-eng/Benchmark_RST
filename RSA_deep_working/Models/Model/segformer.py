import segmentation_models_pytorch as smp
from torch import sigmoid
import torch.nn as nn

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
        in_channels=1, 
        out_channels=1, 
        encoder_name="resnet34", 
        encoder_weights=None,
        decoder_attention_type=None,
        return_logits=False
    ):
        super().__init__()
        self.model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            decoder_attention_type=decoder_attention_type,
            activation=None  # output raw logits, apply activation later
        )
        self.return_logits = return_logits

    def forward(self, x):
        if self.return_logits:
            return self.model(x)
        else: 
            return sigmoid(self.model(x))