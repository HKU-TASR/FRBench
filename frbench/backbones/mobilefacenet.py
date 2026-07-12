"""
MobileFaceNet-ECA backbone model definition.

Reference Papers: 
- MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices (https://arxiv.org/abs/1804.07573)
- ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks (https://arxiv.org/abs/1910.03151)
Reference Code: https://github.com/cavalleria/cavaface/blob/master/backbone/mobilefacenet.py
"""
from typing import List, Tuple
import math

import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm1d,
    BatchNorm2d,
    PReLU,
    Linear,
    AdaptiveAvgPool2d,
)

from .utils import Flatten


__all__ = ["MobileFaceNet", "MobileFaceNet_ECA", "MobileFaceNet_Plain"]


class ECAModule(Module):
    """Efficient Channel Attention (ECA) module.

    Attributes:
        avg_pool: Global average pooling layer.
        conv: 1D convolution for local cross-channel interaction.
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        """Initialize ECA module.
        
        Args:
            channels: Number of input/output channels.
            gamma: Coefficient for adaptive kernel size calculation. Defaults to 2.
            b: Coefficient for adaptive kernel size calculation. Defaults to 1.
        """
        super(ECAModule, self).__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply efficient channel attention to input features.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Attention-recalibrated tensor of shape (N, C, H, W).
        """
        # Global average pooling: (N, C, H, W) -> (N, C, 1, 1)
        y = self.avg_pool(x)
        # Reshape for 1D conv: (N, C, 1, 1) -> (N, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)
        # 1D convolution for local cross-channel interaction
        y = self.conv(y)
        # Reshape back: (N, 1, C) -> (N, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        # Scale input features
        return x * y.expand_as(x)


class ConvBlock(Module):
    """Standard convolution block with Conv-BN-PReLU structure.
    
    Attributes:
        conv: Convolution layer.
        bn: Batch normalization layer.
        prelu: PReLU activation layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        groups: int = 1,
    ) -> None:
        """Initialize ConvBlock.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel. Defaults to (1, 1).
            stride: Convolution stride. Defaults to (1, 1).
            padding: Convolution padding. Defaults to (0, 0).
            groups: Number of groups for grouped convolution. Defaults to 1.
        """
        super(ConvBlock, self).__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = BatchNorm2d(out_channels)
        self.prelu = PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolution block.
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H', W').
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class LinearBlock(Module):
    """Linear convolution block with Conv-BN structure (no activation).
    
    Used for pointwise convolutions in depthwise separable convolutions
    where no activation is needed.
    
    Attributes:
        conv: Convolution layer.
        bn: Batch normalization layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        groups: int = 1,
    ) -> None:
        """Initialize LinearBlock.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel. Defaults to (1, 1).
            stride: Convolution stride. Defaults to (1, 1).
            padding: Convolution padding. Defaults to (0, 0).
            groups: Number of groups for grouped convolution. Defaults to 1.
        """
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear block.
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H', W').
        """
        x = self.conv(x)
        x = self.bn(x)
        return x


class DepthWiseSeparableConv(Module):
    """Depthwise separable conv with optional ECA.

    Attributes:
        residual: Whether to use residual connection.
        use_eca: Whether to apply ECA attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (2, 2),
        padding: Tuple[int, int] = (1, 1),
        groups: int = 1,
        residual: bool = False,
        use_eca: bool = True,
    ) -> None:
        """Initialize DepthWiseSeparableConv.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size for depthwise convolution. Defaults to (3, 3).
            stride: Stride for depthwise convolution. Defaults to (2, 2).
            padding: Padding for depthwise convolution. Defaults to (1, 1).
            groups: Expansion factor for intermediate channels. Defaults to 1.
            residual: Whether to use residual connection. Defaults to False.
            use_eca: Whether to apply ECA attention. Defaults to True.
        """
        super(DepthWiseSeparableConv, self).__init__()
        self.residual = residual
        self.use_eca = use_eca
        
        # 1x1 pointwise expansion
        self.conv_expand = ConvBlock(
            in_channels, groups, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        # 3x3 depthwise convolution
        self.conv_dw = ConvBlock(
            groups, groups, kernel_size=kernel_size, padding=padding, 
            stride=stride, groups=groups
        )
        # 1x1 pointwise projection (linear)
        self.conv_project = LinearBlock(
            groups, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        
        # ECA attention module
        if self.use_eca:
            self.eca = ECAModule(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the depthwise separable convolution.
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H', W').
        """
        identity = x
        
        x = self.conv_expand(x)
        x = self.conv_dw(x)
        x = self.conv_project(x)
        
        if self.use_eca:
            x = self.eca(x)
        
        if self.residual:
            x = x + identity
        
        return x


class ResidualBlock(Module):
    """Residual block containing multiple depthwise separable convolutions.
    
    Stacks multiple depthwise separable convolutions with residual connections.
    
    Attributes:
        num_blocks: Number of depthwise separable convolutions in this block.
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        groups: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 1),
        use_eca: bool = True,
    ) -> None:
        """Initialize ResidualBlock.
        
        Args:
            channels: Number of input/output channels.
            num_blocks: Number of depthwise separable convolutions to stack.
            groups: Expansion factor for intermediate channels.
            kernel_size: Kernel size for depthwise convolution. Defaults to (3, 3).
            stride: Stride for depthwise convolution. Defaults to (1, 1).
            padding: Padding for depthwise convolution. Defaults to (1, 1).
            use_eca: Whether to apply ECA attention. Defaults to True.
        """
        super(ResidualBlock, self).__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(
                DepthWiseSeparableConv(
                    channels, channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    residual=True,
                    use_eca=use_eca,
                )
            )
        self.layers = Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.
        
        Args:
            x: Input tensor of shape (N, channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, channels, H, W).
        """
        return self.layers(x)


class GDC(Module):
    """Global Depthwise Convolution output layer.
    
    Converts spatial feature maps to embedding vectors using depthwise
    convolution followed by fully connected layer with batch normalization.
    
    Attributes:
        num_features: Output embedding dimension.
    """

    def __init__(self, num_features: int = 512) -> None:
        """Initialize GDC output layer.
        
        Args:
            num_features: Output embedding dimension. Defaults to 512.
        """
        super(GDC, self).__init__()
        self.conv_dw = LinearBlock(
            512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(0, 0), groups=512
        )
        self.flatten = Flatten()
        self.fc = Linear(512, num_features, bias=False)
        self.bn = BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert spatial features to embedding vector.
        
        Args:
            x: Input tensor of shape (N, 512, 7, 7).
            
        Returns:
            torch.Tensor: Embedding vector of shape (N, num_features).
        """
        x = self.conv_dw(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.bn(x)
        return x


class GNAP(Module):
    """Global Norm-Aware Pooling output layer.
    
    Alternative output layer that uses norm-aware global average pooling
    for more robust feature aggregation.
    
    Attributes:
        bn1: First batch normalization (without affine).
        pool: Adaptive average pooling.
        bn2: Second batch normalization (without affine).
    """

    def __init__(self, num_features: int = 512) -> None:
        """Initialize GNAP output layer.
        
        Args:
            num_features: Output embedding dimension. Must be 512. Defaults to 512.
        """
        super(GNAP, self).__init__()
        assert num_features == 512, "GNAP only supports num_features=512"
        self.bn1 = BatchNorm2d(512, affine=False)
        self.pool = AdaptiveAvgPool2d((1, 1))
        self.bn2 = BatchNorm1d(512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert spatial features to embedding vector using norm-aware pooling.
        
        Args:
            x: Input tensor of shape (N, 512, H, W).
            
        Returns:
            torch.Tensor: Embedding vector of shape (N, 512).
        """
        x = self.bn1(x)
        # Compute L2 norm for each spatial position
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        # Norm-aware weighting
        weight = x_norm_mean / x_norm
        x = x * weight
        # Global average pooling
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.bn2(x)
        return x


class MobileFaceNet(Module):
    """MobileFaceNet backbone with optional ECA attention.

    Attributes:
        num_features: Output embedding dimension.
        use_eca: Whether ECA attention is enabled.
        output_name: Type of output layer ('GDC' or 'GNAP').
    """

    def __init__(
        self,
        input_size: List[int] = [112, 112],
        num_features: int = 512,
        output_name: str = "GDC",
        use_eca: bool = False,
    ) -> None:
        """Initialize MobileFaceNet backbone.
        
        Args:
            input_size: Input image size as [H, W]. Only [112, 112] is supported.
                Defaults to [112, 112].
            num_features: Output embedding dimension. Defaults to 512.
            output_name: Output layer type, either 'GDC' (Global Depthwise Conv)
                or 'GNAP' (Global Norm-Aware Pooling). Defaults to 'GDC'.
            use_eca: Whether to use ECA attention modules. Defaults to False.
        """
        super(MobileFaceNet, self).__init__()
        assert output_name in ["GNAP", "GDC"], "output_name must be 'GNAP' or 'GDC'"
        assert input_size[0] == 112, "Only input_size [112, 112] is supported"
        
        self.num_features = num_features
        self.use_eca = use_eca
        self.output_name = output_name
        
        # Stage 1: Initial convolution
        # 112x112 -> 56x56
        self.conv1 = ConvBlock(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64
        )
        
        # Stage 2: First depthwise separable block
        # 56x56 -> 28x28
        self.conv_23 = DepthWiseSeparableConv(
            64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
            groups=128, residual=False, use_eca=use_eca
        )
        self.conv_3 = ResidualBlock(
            64, num_blocks=4, groups=128, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), use_eca=use_eca
        )
        
        # Stage 3: Second depthwise separable block
        # 28x28 -> 14x14
        self.conv_34 = DepthWiseSeparableConv(
            64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
            groups=256, residual=False, use_eca=use_eca
        )
        self.conv_4 = ResidualBlock(
            128, num_blocks=6, groups=256, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), use_eca=use_eca
        )
        
        # Stage 4: Third depthwise separable block
        # 14x14 -> 7x7
        self.conv_45 = DepthWiseSeparableConv(
            128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
            groups=512, residual=False, use_eca=use_eca
        )
        self.conv_5 = ResidualBlock(
            128, num_blocks=2, groups=256, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), use_eca=use_eca
        )
        
        # Stage 5: Final 1x1 convolution to expand channels
        self.conv_6_sep = ConvBlock(
            128, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        
        # Output layer
        if output_name == "GNAP":
            self.output_layer = GNAP(num_features)
        else:
            self.output_layer = GDC(num_features)
        
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embeddings from input images.
        
        Args:
            x: Input images of shape (N, 3, 112, 112).
            
        Returns:
            torch.Tensor: Face embeddings of shape (N, num_features).
        """
        # Stage 1
        x = self.conv1(x)
        x = self.conv2_dw(x)
        
        # Stage 2
        x = self.conv_23(x)
        x = self.conv_3(x)
        
        # Stage 3
        x = self.conv_34(x)
        x = self.conv_4(x)
        
        # Stage 4
        x = self.conv_45(x)
        x = self.conv_5(x)
        
        # Stage 5
        x = self.conv_6_sep(x)
        
        # Output
        x = self.output_layer(x)
        
        return x


def MobileFaceNet_ECA(**kwargs) -> MobileFaceNet:
    """Construct a MobileFaceNet model with ECA attention.

    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        output_name (str): Output layer type ('GDC' or 'GNAP'). Defaults to 'GDC'.
        
    Returns:
        MobileFaceNet: MobileFaceNet-ECA model instance.
    """
    return MobileFaceNet(use_eca=True, **kwargs)


def MobileFaceNet_Plain(**kwargs) -> MobileFaceNet:
    """Construct a plain MobileFaceNet model (no ECA).

    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        output_name (str): Output layer type ('GDC' or 'GNAP'). Defaults to 'GDC'.
        
    Returns:
        MobileFaceNet: Plain MobileFaceNet model instance.
    """
    return MobileFaceNet(use_eca=False, **kwargs)