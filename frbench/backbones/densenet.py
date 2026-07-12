"""
DenseNet backbone for face recognition.

Reference Paper: Densely Connected Convolutional Networks (https://arxiv.org/abs/1608.06993)
Reference Code: https://github.com/cavalleria/cavaface/blob/master/backbone/densenet.py
"""
from typing import List

import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm1d,
    BatchNorm2d,
    PReLU,
    Dropout,
    AvgPool2d,
    Linear,
)

from .utils import Flatten


__all__ = ["DenseNet", "DenseNet_121", "DenseNet_161", "DenseNet_169", "DenseNet_201"]


class BNPReLUConv(Module):
    """Pre-activated convolution block with BatchNorm-PReLU-Conv structure.
    
    Attributes:
        bn: Batch normalization layer.
        activ: PReLU activation layer.
        conv: Convolution layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        """Initialize BNPReLUConv block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel.
            stride: Convolution stride. Defaults to 1.
            padding: Convolution padding. Defaults to 0.
        """
        super(BNPReLUConv, self).__init__()
        self.bn = BatchNorm2d(num_features=in_channels)
        self.activ = PReLU(in_channels)
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the pre-activated convolution block.
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H', W').
        """
        x = self.bn(x)
        x = self.activ(x)
        x = self.conv(x)
        return x


class DenseLayer(Module):
    """Dense layer (bottleneck) that produces growth_rate new feature maps.
    
    Attributes:
        bn_size: Bottleneck width multiplier (typically 4).
        growth_rate: Number of new feature maps produced by this layer.
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize DenseLayer.
        
        Args:
            in_channels: Number of input channels.
            growth_rate: Number of output feature maps (k in the paper).
            bn_size: Bottleneck size multiplier. Defaults to 4.
            dropout_rate: Dropout probability. Defaults to 0.0.
        """
        super(DenseLayer, self).__init__()
        mid_channels = bn_size * growth_rate
        
        self.conv1 = BNPReLUConv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
        )
        self.conv2 = BNPReLUConv(
            in_channels=mid_channels,
            out_channels=growth_rate,
            kernel_size=3,
            padding=1,
        )
        self.use_dropout = dropout_rate > 0.0
        if self.use_dropout:
            self.dropout = Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dense layer.
        
        Concatenates input with new features (dense connectivity).
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, in_channels + growth_rate, H, W).
        """
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = torch.cat([identity, x], dim=1)
        return x


class DenseBlock(Module):
    """Dense block containing multiple dense layers with dense connectivity.
    
    Attributes:
        num_layers: Number of dense layers in this block.
        growth_rate: Number of new channels added by each layer.
    """

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize DenseBlock.
        
        Args:
            num_layers: Number of dense layers in the block.
            in_channels: Number of input channels to the block.
            growth_rate: Number of output channels per dense layer.
            bn_size: Bottleneck size multiplier. Defaults to 4.
            dropout_rate: Dropout probability. Defaults to 0.0.
        """
        super(DenseBlock, self).__init__()
        self.layers = Sequential()
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels=in_channels + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout_rate=dropout_rate,
            )
            self.layers.add_module(f"denselayer{i + 1}", layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all dense layers in the block.
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor with accumulated feature maps.
        """
        return self.layers(x)


class Transition(Module):
    """Transition layer between dense blocks.
    
    Attributes:
        compression: Channel reduction factor (theta in the paper).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize Transition layer.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (typically in_channels // 2).
        """
        super(Transition, self).__init__()
        self.conv = BNPReLUConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.pool = AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transition layer.
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H//2, W//2).
        """
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet(Module):
    """DenseNet backbone for face recognition.
    
    Attributes:
        num_features: Output embedding dimension.
        growth_rate: Number of new channels added per dense layer.
    """

    def __init__(
        self,
        layers: List[int],
        growth_rate: int = 32,
        init_channels: int = 64,
        bn_size: int = 4,
        compression: float = 0.5,
        input_size: List[int] = [112, 112],
        num_features: int = 512,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize DenseNet backbone.
        
        Args:
            layers: Number of dense layers in each of the 4 dense blocks.
            growth_rate: Number of filters added per dense layer (k). Defaults to 32.
            init_channels: Number of channels in the initial convolution. Defaults to 64.
            bn_size: Bottleneck width multiplier. Defaults to 4.
            compression: Compression factor at transition layers. Defaults to 0.5.
            input_size: Input image size as [H, W]. Defaults to [112, 112].
            num_features: Output embedding dimension. Defaults to 512.
            dropout_rate: Dropout probability in dense layers. Defaults to 0.0.
        """
        super(DenseNet, self).__init__()
        assert input_size[0] in [112], "input_size should be [112, 112]"
        
        self.num_features = num_features
        self.growth_rate = growth_rate
        
        # Initial convolution layer
        self.input_layer = Sequential(
            Conv2d(3, init_channels, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(init_channels),
            PReLU(init_channels),
        )
        
        # Build dense blocks and transition layers
        self.features = Sequential()
        num_channels = init_channels
        
        for i, num_layers in enumerate(layers):
            # Add dense block
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout_rate=dropout_rate,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_channels = num_channels + num_layers * growth_rate
            
            # Add transition layer (except after the last dense block)
            if i < len(layers) - 1:
                out_channels = int(num_channels * compression)
                trans = Transition(in_channels=num_channels, out_channels=out_channels)
                self.features.add_module(f"transition{i + 1}", trans)
                num_channels = out_channels
        
        # Final batch normalization
        self.features.add_module("norm_final", BatchNorm2d(num_channels))
        
        # Output layers
        # For 112x112 input: after initial conv (56x56) and 3 transitions (7x7)
        self.output_layer = Sequential(
            Dropout(p=0.4),
            Flatten(),
            Linear(num_channels * 7 * 7, num_features),
            BatchNorm1d(num_features),
        )
        
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embeddings from input images.
        
        Args:
            x: Input images of shape (N, 3, H, W).
            
        Returns:
            torch.Tensor: Face embeddings of shape (N, num_features).
        """
        x = self.input_layer(x)
        x = self.features(x)
        x = self.output_layer(x)
        return x


def DenseNet_121(**kwargs) -> DenseNet:
    """Construct a DenseNet-121 model.
    
    DenseNet-121 has 121 layers with configuration [6, 12, 24, 16].
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.0.
        
    Returns:
        DenseNet: DenseNet-121 model instance.
    """
    return DenseNet(layers=[6, 12, 24, 16], growth_rate=32, init_channels=64, **kwargs)


def DenseNet_161(**kwargs) -> DenseNet:
    """Construct a DenseNet-161 model.
    
    DenseNet-161 has 161 layers with configuration [6, 12, 36, 24] and larger
    growth rate of 48.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.0.
        
    Returns:
        DenseNet: DenseNet-161 model instance.
    """
    return DenseNet(layers=[6, 12, 36, 24], growth_rate=48, init_channels=96, **kwargs)


def DenseNet_169(**kwargs) -> DenseNet:
    """Construct a DenseNet-169 model.
    
    DenseNet-169 has 169 layers with configuration [6, 12, 32, 32].
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.0.
        
    Returns:
        DenseNet: DenseNet-169 model instance.
    """
    return DenseNet(layers=[6, 12, 32, 32], growth_rate=32, init_channels=64, **kwargs)


def DenseNet_201(**kwargs) -> DenseNet:
    """Construct a DenseNet-201 model.
    
    DenseNet-201 has 201 layers with configuration [6, 12, 48, 32].
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.0.
        
    Returns:
        DenseNet: DenseNet-201 model instance.
    """
    return DenseNet(layers=[6, 12, 48, 32], growth_rate=32, init_channels=64, **kwargs)

