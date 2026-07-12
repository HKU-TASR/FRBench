from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout,
    Sequential,
    BatchNorm1d,
    BatchNorm2d,
    GroupNorm,
    LayerNorm,
    Identity,
)


class Flatten(Module):
    """Flatten module that reshapes tensor to (batch_size, -1)."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten the input tensor.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            Flattened tensor of shape (N, C*H*W).
        """
        return x.view(x.size(0), -1)


class DropPath(Module):
    """Drop paths (Stochastic Depth) per sample.
    
    Randomly drops entire residual branches during training for regularization.
    
    Attributes:
        drop_prob: Probability of dropping a path.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        """Initialize DropPath module.
        
        Args:
            drop_prob: Probability of dropping the path. Defaults to 0.0.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply drop path to input tensor.
        
        Args:
            x: Input tensor of any shape.
            
        Returns:
            Output tensor with same shape, paths dropped during training.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with arbitrary dimensions
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output


def trunc_normal_(tensor: torch.Tensor, mean: float = 0., std: float = 1.) -> torch.Tensor:
    """Truncated normal initialization.
    
    Args:
        tensor: Tensor to initialize.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
        
    Returns:
        Initialized tensor.
    """
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
    return tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> Conv2d:
    """Create a 3x3 convolution layer with padding.
    
    Args:
        in_planes: Number of input channels.
        out_planes: Number of output channels.
        stride: Convolution stride. Defaults to 1.
        
    Returns:
        A Conv2d layer with kernel_size=3, padding=1, bias=False.
    """
    return Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> Conv2d:
    """Create a 1x1 convolution layer (pointwise convolution).
    
    Args:
        in_planes: Number of input channels.
        out_planes: Number of output channels.
        stride: Convolution stride. Defaults to 1.
        
    Returns:
        A Conv2d layer with kernel_size=1, bias=False.
    """
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class LayerNorm2d(Module):
    """Channel-wise LayerNorm over NCHW feature maps (for the FR head).

    Attributes:
        weight: Learnable per-channel scale.
        bias: Learnable per-channel shift.
        eps: Numerical-stability constant.
        normalized_shape: The (C,) shape normalized over.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        """Initialize LayerNorm2d.

        Args:
            normalized_shape: Number of channels to normalize.
            eps: Numerical-stability constant. Defaults to 1e-6.
        """
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise LayerNorm.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Normalized tensor of shape (B, C, H, W), made contiguous so the
            head's custom ``Flatten`` (which calls ``.view``) is safe.
        """
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


def build_fr_head(
    in_channels: int,
    spatial_size: Tuple[int, int],
    num_features: int,
    dropout_rate: float = 0.0,
    feat_bn: bool = True,
) -> Sequential:
    """Build the FaceX-Zoo-style position-preserving embedding head.

    Args:
        in_channels: Channels of the final feature map.
        spatial_size: ``(H, W)`` of the final feature map.
        num_features: Output embedding dimension.
        dropout_rate: Dropout before the linear projection. Defaults to 0.0.
        feat_bn: Whether to append a BatchNorm1d (BN-Neck). Defaults to True.

    Returns:
        A ``Sequential`` implementing the head.
    """
    flat_dim = spatial_size[0] * spatial_size[1] * in_channels
    return Sequential(
        LayerNorm2d(in_channels, eps=1e-6),
        Flatten(),
        Dropout(p=dropout_rate) if dropout_rate > 0 else Identity(),
        Linear(flat_dim, num_features),
        BatchNorm1d(num_features) if feat_bn else Identity(),
    )


def infer_output_size(input_size: List[int], num_downsamples: int = 5) -> Tuple[int, int]:
    """Compute the final feature-map size after a stack of stride-2 stages.

    Args:
        input_size: ``[H, W]`` of the input image.
        num_downsamples: Number of stride-2 stages. Defaults to 5.

    Returns:
        ``(H_final, W_final)`` of the last feature map.
    """
    h, w = input_size[0], input_size[1]
    for _ in range(num_downsamples):
        h = (h + 1) // 2
        w = (w + 1) // 2
    assert h > 0 and w > 0, (
        f"Final-stage resolution must be positive, got ({h}, {w}). "
        f"Check that input_size produces enough downsampling stages."
    )
    return (h, w)


def init_weights(m: Module) -> None:
    """Initialize a module's parameters (called via ``self.apply``).

    Args:
        m: Module to initialize.
    """
    if isinstance(m, Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (BatchNorm1d, BatchNorm2d, GroupNorm, LayerNorm, LayerNorm2d)):
        if getattr(m, "weight", None) is not None:
            nn.init.ones_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)