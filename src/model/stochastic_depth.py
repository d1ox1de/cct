import torch
import torch.nn as nn

from typing import Optional

class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor):
        return drop_path(x, self.drop_prob, self.training)


def drop_path(x: torch.Tensor, drop_prob: float = 0., is_training: bool = False) -> torch.Tensor:
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... 
    """

    if drop_prob == 0. or not is_training:
        return x
    
    keep_prob = 1. - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.dim() - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob
    random_tensor.floor_() # binarize
    output = x.div(keep_prob) * random_tensor
    return output

