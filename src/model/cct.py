import torch
import torch.nn as nn

from pathlib import Path
from typing import Optional

from .transformer import TransformerClassifier
from .tokenizer import Tokenizer
from .utils import pe_check, fc_check

class CCT(nn.Module):
    
    def __init__(self,
                embed_dim: int = 768,
                img_size: int = 224,
                n_input_channels: int = 3,
                n_conv_layers: int = 1,
                kernel_size: int = 7,
                stride: int = 2,
                padding: int = 3,
                pooling_kernel_size: int = 3,
                pooling_stride:int = 2,
                pooling_padding: int = 1,
                dropout: float = 0.,
                attention_dropout: float = 0.1,
                stochastic_depth: float = 0.1,
                num_layers: int = 14,
                num_heads: int = 6,
                mlp_ratio: float = 4.0,
                num_classes: int = 10,
                positional_embedding: str = 'learnable'
                ):
        super().__init__()
        
        self.tokenizer = Tokenizer(
            kernel_size=kernel_size,
            stride=stride, padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride, 
            pooling_padding=pooling_padding,
            n_conv_layers=n_conv_layers, 
            n_input_channels=n_input_channels, 
            n_output_channels=embed_dim,
            activation=nn.ReLU, 
            is_max_pool=True, 
            is_conv_bias=False)
        
        sequence_length = self.tokenizer.sequence_length(n_channels=n_input_channels, height=img_size, width=img_size)
        self.classifier = TransformerClassifier(
            num_classes=num_classes, 
            embed_dims=embed_dim, 
            num_layers=num_layers,
            num_heads=num_heads, 
            fc_ration=mlp_ratio, 
            dropout=dropout,
            attention_dropout=attention_dropout, 
            stochastic_depth=stochastic_depth, 
            positional_embed=positional_embedding,
            sequence_pool=True, 
            sequence_length=sequence_length)
    
    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


def _cct(
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        embedding_dim: int, 
        weights: Optional[str] = None,
        kernel_size: int = 3, 
        stride=None, 
        padding=None,  
        positional_embedding='learnable',
        *args, **kwargs):

    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    
    model = CCT(num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embed_dim=embedding_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                *args, **kwargs)

    if weights is not None:
        print('Loading pretrained weights')
        weights = Path(weights)
        state_dict = torch.load(str(weights)) # for other scenarios it would be beneficial to use map_location
        if positional_embedding == 'learnable':
            state_dict = pe_check(model, state_dict)
        state_dict = fc_check(model, state_dict)
        missing_keys = model.load_state_dict(state_dict, strict=True)
    print(missing_keys)
    
    return model


def cct(
        weights: Optional[str],
        num_classes: int = 1000,
        img_size: int = 224,
        *args,
        **kwargs):

    model = _cct(
        weights=weights,
        img_size=img_size,
        num_classes=num_classes,
        kernel_size=7,
        n_conv_layers=2,
        num_layers=14,
        num_heads=6,
        mlp_ratio=3,
        embedding_dim=384,
        *args,
        **kwargs
    )
    
    return model