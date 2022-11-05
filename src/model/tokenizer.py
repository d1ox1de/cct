import torch
import torch.nn as nn

from typing import Callable, Optional

class Tokenizer(nn.Module):

    def __init__(self,
                kernel_size: int, stride: int, padding: int,
                pooling_kernel_size: int = 3, pooling_stride: int = 2, pooling_padding: int = 1,
                n_conv_layers: int = 1, n_input_channels: int = 3, n_output_channels: int = 64,
                activation: Optional[Callable] = None, is_max_pool: bool = True, is_conv_bias: bool = False,
                in_planes: int = 64
                ):

        super().__init__()
        
        n_filter_ls = [n_input_channels] + [in_planes for _ in range(n_conv_layers - 1)] + [n_output_channels]
        
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(
                    in_channels=n_filter_ls[i], out_channels=n_filter_ls[i + 1],
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, bias=is_conv_bias),
                
                activation() if activation is not None else nn.Identity(),
                
                nn.MaxPool2d(
                    kernel_size=pooling_kernel_size,
                    stride=pooling_stride,
                    padding=pooling_padding) if is_max_pool else nn.Identity())
            for i in range(n_conv_layers)]
            )

        self.flatten = nn.Flatten(2, 3)
        # Applies ``fn`` recursively to every submodule
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x ~ (B, C_in=3, H, W)
        return self.flatten(self.conv_layers(x)).transpose(1, 2)


    def sequence_length(self, n_channels: int = 3, height: int = 224, width: int = 224) -> int:
        return self.forward(torch.zeros(1, n_channels, height, width)).shape[1]


    @staticmethod
    def _init_weights(submodule):
        if isinstance(submodule, nn.Conv2d):
            nn.init.kaiming_normal_(submodule.weight)
