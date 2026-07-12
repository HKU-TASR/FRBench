"""
MobileNet backbone for face recognition.

Reference Paper:
    - MobileNet: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (https://arxiv.org/abs/1704.04861)
    - MobileNetV2: MobileNetV2: Inverted Residuals and Linear Bottlenecks (https://arxiv.org/abs/1801.04381)
    - MobileNetV3: Searching for MobileNetV3 (https://arxiv.org/abs/1905.02244)
    - MobileNetV4: MobileNetV4 -- Universal Models for the Mobile Ecosystem (https://arxiv.org/abs/2404.10518)
Reference Code: 
    - MobileNet:https://github.com/deepinsight/insightface/blob/master/recognition/arcface_mxnet/symbol/fmobilenet.py
    - MobileNetV2: https://github.com/cavalleria/cavaface/blob/master/backbone/mobilenetv2.py
    - MobileNetV3: https://github.com/cavalleria/cavaface/blob/master/backbone/mobilenetv3.py
    - MobileNetV4: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv3.py
Supported Variants: 
    - MobileNet-1.0, MobileNet-0.75, MobileNet-0.5, MobileNet-0.25
    - MobileNetV2-1.0, MobileNetV2-0.75, MobileNetV2-0.5, MobileNetV2-0.25
    - MobileNetV3-Large, MobileNetV3-Small
    - MobileNetV4-Conv-Small, MobileNetV4-Conv-Medium, MobileNetV4-Conv-Large
    - MobileNetV4-Hybrid-Medium, MobileNetV4-Hybrid-Large
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm1d,
    BatchNorm2d,
    ReLU,
    Dropout,
    Linear,
    AdaptiveAvgPool2d,
)


__all__ = [
    "MobileNet",
    "MobileNet_W1",
    "MobileNet_W3D4",
    "MobileNet_WD2",
    "MobileNet_WD4",
    "MobileNetV2",
    "MobileNetV2_W1",
    "MobileNetV2_W3D4",
    "MobileNetV2_WD2",
    "MobileNetV2_WD4",
    "MobileNetV3",
    "MobileNetV3_Large",
    "MobileNetV3_Small",
    "MobileNetV4",
    "MobileNetV4_Conv_Small",
    "MobileNetV4_Conv_Medium",
    "MobileNetV4_Conv_Large",
    "MobileNetV4_Hybrid_Medium",
    "MobileNetV4_Hybrid_Large",
]


class ConvBNPReLU_V1(Module):
    """Conv-BN(fix_gamma=True)-PReLU block for MobileNetV1.

    Attributes:
        conv: Convolution layer.
        bn: Batch normalization layer with fixed gamma.
        act: PReLU activation layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ) -> None:
        """Initialize ConvBNPReLU_V1.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel. Defaults to 3.
            stride: Convolution stride. Defaults to 1.
            padding: Convolution padding. Defaults to 1.
            groups: Number of groups for grouped convolution. Defaults to 1.
        """
        super(ConvBNPReLU_V1, self).__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        # fix_gamma=True: BN weight (gamma) fixed to 1.0, bias (beta) still learnable
        self.bn = BatchNorm2d(out_channels)
        nn.init.constant_(self.bn.weight, 1.0)
        self.bn.weight.requires_grad = False
        # PReLU activation to match InsightFace's fmobilenet.py
        self.act = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolution block.
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H', W').
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DepthwiseSeparableConv(Module):
    """Depthwise separable convolution: 3x3 depthwise + 1x1 pointwise.

    Attributes:
        depthwise: Depthwise convolution block (Conv-BN-ReLU).
        pointwise: Pointwise convolution block (Conv-BN-ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        """Initialize DepthwiseSeparableConv.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for depthwise convolution. Defaults to 1.
        """
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution: 3x3 conv with groups=in_channels
        self.depthwise = ConvBNPReLU_V1(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
        )
        # Pointwise convolution: 1x1 conv
        self.pointwise = ConvBNPReLU_V1(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the depthwise separable convolution.
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H', W').
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNet(Module):
    """MobileNet backbone for face recognition.

    Attributes:
        num_features: Output embedding dimension.
        width_mult: Width multiplier for channel scaling.
    """

    def __init__(
        self,
        width_mult: float = 1.0,
        input_size: List[int] = [112, 112],
        num_features: int = 512,
        dropout_rate: float = 0.4,
        round_channels: bool = True,
    ) -> None:
        """Initialize MobileNet backbone.
        
        Args:
            width_mult: Width multiplier for scaling channel dimensions.
                Common values are 1.0, 0.75, 0.5, 0.25. Defaults to 1.0.
            input_size: Input image size as [H, W]. Supports [112, 112] 
                or [224, 224]. Defaults to [112, 112].
            num_features: Output embedding dimension. Defaults to 512.
            dropout_rate: Unused (kept for config compatibility).
            round_channels: Round channel counts to divisible by 8. Defaults to True.
        """
        super(MobileNet, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        
        self.num_features = num_features
        self.width_mult = width_mult
        
        def _make_divisible(v: float, divisor: int = 8) -> int:
            """Make channel count divisible by divisor for hardware efficiency."""
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            # Ensure round down doesn't reduce by more than 10%
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        def _scale_channels(v: float) -> int:
            """Scale channel count by width multiplier."""
            if round_channels:
                return _make_divisible(v * width_mult)
            else:
                return int(v * width_mult)
        
        # Channel configuration for each layer
        # Format: (out_channels, stride)
        # Original MobileNet for ImageNet uses these base channels:
        # Conv: 32 -> DW/PW stages with increasing channels
        base_channels = [
            (64, 1),    # Stage 1: 1 layer
            (128, 2),   # Stage 2: 2 layers (first with stride 2)
            (128, 1),
            (256, 2),   # Stage 3: 2 layers (first with stride 2)
            (256, 1),
            (512, 2),   # Stage 4: 6 layers (first with stride 2)
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (1024, 2),  # Stage 5: 2 layers (first with stride 2)
            (1024, 1),
        ]
        
        # Apply width multiplier
        init_channels = _scale_channels(32)
        channels = [((_scale_channels(c), s)) for c, s in base_channels]
        
        # Build feature extraction layers
        self.features = Sequential()
        
        # Initial 3x3 convolution layer
        init_stride = 1 if input_size[0] == 112 else 2
        self.features.add_module(
            "init_block",
            ConvBNPReLU_V1(3, init_channels, kernel_size=3, stride=init_stride, padding=1)
        )
        
        # Depthwise separable convolution layers
        in_channels = init_channels
        for i, (out_channels, stride) in enumerate(channels):
            self.features.add_module(
                f"dw_block_{i + 1}",
                DepthwiseSeparableConv(in_channels, out_channels, stride)
            )
            in_channels = out_channels
        
        # Output layers (GDC head matching InsightFace's get_fc1 GDC type)
        # GDC = Linear(depthwise 7x7) -> Flatten -> FC -> BN(fix_gamma=True)
        # No dropout in InsightFace's GDC head
        final_channels = _scale_channels(1024)
        self.gdc_conv = Conv2d(
            final_channels, final_channels,
            kernel_size=7, stride=1, padding=0,
            groups=final_channels, bias=False,
        )
        self.gdc_bn = BatchNorm2d(final_channels)
        
        self.fc = Linear(final_channels, num_features, bias=False)
        self.bn = BatchNorm1d(num_features, eps=1e-05)
        
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights using Kaiming initialization.
        
        """
        for name, m in self.named_modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (BatchNorm2d, BatchNorm1d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embeddings from input images.
        
        Args:
            x: Input images of shape (N, 3, H, W).
            
        Returns:
            torch.Tensor: Face embeddings of shape (N, num_features).
        """
        # Feature extraction
        x = self.features(x)
        
        # GDC output layer (replaces GAP)
        x = self.gdc_conv(x)
        x = self.gdc_bn(x)
        
        # Flatten and output layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        
        return x


def MobileNet_W1(**kwargs) -> MobileNet:
    """Construct a MobileNet-1.0 model (full width).
    
    Standard MobileNet with width multiplier 1.0, providing the best
    accuracy among MobileNet variants.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        
    Returns:
        MobileNet: MobileNet-1.0 model instance.
    """
    return MobileNet(width_mult=1.0, **kwargs)


def MobileNet_W3D4(**kwargs) -> MobileNet:
    """Construct a MobileNet-0.75 model (3/4 width).
    
    MobileNet with width multiplier 0.75, offering a balance between
    accuracy and computational efficiency.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        
    Returns:
        MobileNet: MobileNet-0.75 model instance.
    """
    return MobileNet(width_mult=0.75, **kwargs)


def MobileNet_WD2(**kwargs) -> MobileNet:
    """Construct a MobileNet-0.5 model (half width).
    
    MobileNet with width multiplier 0.5, providing significant computational
    savings with moderate accuracy reduction.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        
    Returns:
        MobileNet: MobileNet-0.5 model instance.
    """
    return MobileNet(width_mult=0.5, **kwargs)


def MobileNet_WD4(**kwargs) -> MobileNet:
    """Construct a MobileNet-0.25 model (quarter width).
    
    MobileNet with width multiplier 0.25, the most lightweight variant
    suitable for extremely resource-constrained environments.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        
    Returns:
        MobileNet: MobileNet-0.25 model instance.
    """
    return MobileNet(width_mult=0.25, **kwargs)


# =============================================================================
# MobileNetV2 Implementation
# =============================================================================


class ConvBNPReLU(Module):
    """Convolution block with Conv-BN-PReLU structure for MobileNetV2.
    
    This implementation follows cavaface's MobileNetV2 which uses PReLU
    activation instead of ReLU6. PReLU typically performs better for
    face recognition tasks.
    
    Attributes:
        conv: Convolution layer.
        bn: Batch normalization layer.
        prelu: PReLU activation layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        """Initialize ConvBNPReLU.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel. Defaults to 3.
            stride: Convolution stride. Defaults to 1.
            groups: Number of groups for grouped convolution. Defaults to 1.
        """
        super(ConvBNPReLU, self).__init__()
        padding = (kernel_size - 1) // 2
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
        self.prelu = nn.PReLU(out_channels)

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


class InvertedResidual(Module):
    """Inverted residual block for MobileNetV2 (cavaface: PReLU).

    Attributes:
        use_residual: Whether to use residual connection.
        conv: Sequential convolution layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        """Initialize InvertedResidual block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for depthwise convolution (1 or 2).
            expand_ratio: Expansion factor for hidden dimension.
        """
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2], "Stride must be 1 or 2"
        
        self.stride = stride
        self.use_residual = (stride == 1) and (in_channels == out_channels)
        
        hidden_channels = int(round(in_channels * expand_ratio))
        
        layers = []
        
        # Expansion phase: 1x1 conv to expand channels (skip if expand_ratio == 1)
        if expand_ratio != 1:
            layers.append(
                ConvBNPReLU(in_channels, hidden_channels, kernel_size=1)
            )
        
        # Depthwise phase: 3x3 depthwise conv
        layers.append(
            ConvBNPReLU(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                groups=hidden_channels,
            )
        )
        
        # Projection phase: 1x1 conv to project back (LINEAR - no activation!)
        layers.extend([
            Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm2d(out_channels),
        ])
        
        self.conv = Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the inverted residual block.
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H', W').
        """
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(Module):
    """MobileNetV2 backbone for face recognition (cavaface variant).

    Attributes:
        num_features: Output embedding dimension.
        width_mult: Width multiplier for channel scaling.
    """

    def __init__(
        self,
        width_mult: float = 1.0,
        input_size: List[int] = [112, 112],
        num_features: int = 512,
    ) -> None:
        """Initialize MobileNetV2 backbone.
        
        Args:
            width_mult: Width multiplier for scaling channel dimensions.
                Common values are 1.0, 0.75, 0.5, 0.25. Defaults to 1.0.
            input_size: Input image size as [H, W]. Only [112, 112] is officially
                supported. Other sizes may cause issues with GDC layer.
                Defaults to [112, 112].
            num_features: Output embedding dimension. Defaults to 512.
        """
        super(MobileNetV2, self).__init__()
        assert input_size[0] in [112], "input_size should be [112, 112]. Other sizes may cause GDC layer issues."
        
        self.num_features = num_features
        self.width_mult = width_mult
        
        round_nearest = 8
        
        def _make_divisible(v: float, divisor: int = 8) -> int:
            """Make channel count divisible by divisor for hardware efficiency.
            
            This function is taken from the original TensorFlow implementation.
            It ensures that all layers have a channel number divisible by 8.
            """
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            # Ensure round down doesn't reduce by more than 10%
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        inverted_residual_setting = [
            # t, c,  n, s
            [1, 16,  1, 1],   # Stage 1
            # [6, 24,  2, 2], # Skipped in cavaface version
            [6, 32,  3, 2],   # Stage 2
            [6, 64,  4, 2],   # Stage 3
            [6, 96,  3, 1],   # Stage 4
            [6, 160, 3, 2],   # Stage 5
            [6, 320, 1, 1],   # Stage 6
        ]
        
        # Initial channel count
        input_channel = _make_divisible(32 * width_mult, round_nearest)
        
        # Last channel count (cavaface uses 512 instead of 1280)
        last_channel = _make_divisible(512 * max(1.0, width_mult), round_nearest)
        
        # Build feature extraction layers
        features = []
        
        # First layer: standard 3x3 conv with stride 2
        features.append(ConvBNPReLU(3, input_channel, kernel_size=3, stride=2))
        
        # Inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel
        
        # Last convolution layer: 1x1 conv to expand to last_channel
        features.append(ConvBNPReLU(input_channel, last_channel, kernel_size=1))
        
        self.features = Sequential(*features)
        
        # Output layer: Global Depthwise Convolution (GDC)
        feature_map_size = 7
        self.output_layer = GDC(
            in_channels=last_channel,
            num_features=num_features,
            input_size=feature_map_size,
        )
        
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights.
        
        Follows the initialization scheme from cavaface:
        - Conv2d: Kaiming normal initialization (fan_out mode)
        - BatchNorm: weight=1, bias=0
        - Linear: Kaiming normal initialization
        """
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (BatchNorm2d, BatchNorm1d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embeddings from input images.
        
        Args:
            x: Input images of shape (N, 3, H, W).
            
        Returns:
            torch.Tensor: Face embeddings of shape (N, num_features).
        """
        x = self.features(x)
        x = self.output_layer(x)
        return x


def MobileNetV2_W1(**kwargs) -> MobileNetV2:
    """Construct a MobileNetV2 model.

    Keyword Args:
        input_size (List[int]): Input image size. Only [112, 112] is supported.
            Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        
    Returns:
        MobileNetV2: MobileNetV2-1.0 model instance.
    """
    return MobileNetV2(width_mult=1.0, **kwargs)


def MobileNetV2_W3D4(**kwargs) -> MobileNetV2:
    """Construct a MobileNetV2-0.75 model (3/4 width).
    
    MobileNetV2 with width multiplier 0.75, offering a balance between
    accuracy and computational efficiency.
    
    Note:
        Only supports 112x112 input. Using other input sizes may cause
        issues with the GDC output layer.
    
    Keyword Args:
        input_size (List[int]): Input image size. Only [112, 112] is supported.
            Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        
    Returns:
        MobileNetV2: MobileNetV2-0.75 model instance.
    """
    return MobileNetV2(width_mult=0.75, **kwargs)


def MobileNetV2_WD2(**kwargs) -> MobileNetV2:
    """Construct a MobileNetV2-0.5 model (half width).
    
    MobileNetV2 with width multiplier 0.5, providing significant computational
    savings with moderate accuracy reduction.
    
    Note:
        Only supports 112x112 input. Using other input sizes may cause
        issues with the GDC output layer.
    
    Keyword Args:
        input_size (List[int]): Input image size. Only [112, 112] is supported.
            Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        
    Returns:
        MobileNetV2: MobileNetV2-0.5 model instance.
    """
    return MobileNetV2(width_mult=0.5, **kwargs)


def MobileNetV2_WD4(**kwargs) -> MobileNetV2:
    """Construct a MobileNetV2-0.25 model (quarter width).
    
    MobileNetV2 with width multiplier 0.25, the most lightweight variant
    suitable for extremely resource-constrained environments.
    
    Note:
        Only supports 112x112 input. Using other input sizes may cause
        issues with the GDC output layer.
    
    Keyword Args:
        input_size (List[int]): Input image size. Only [112, 112] is supported.
            Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        
    Returns:
        MobileNetV2: MobileNetV2-0.25 model instance.
    """
    return MobileNetV2(width_mult=0.25, **kwargs)


# =============================================================================
# MobileNetV3 Implementation
# =============================================================================


def _make_divisible(v: float, divisor: int = 8, min_value: int = None) -> int:
    """Make channel count divisible by divisor for hardware efficiency.
    
    This function is taken from the original TensorFlow implementation.
    It ensures that all layers have a channel number that is divisible by 8.
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    
    Args:
        v: Original number of channels.
        divisor: Alignment value. Defaults to 8.
        min_value: Minimum value threshold. Defaults to divisor.
        
    Returns:
        int: Rounded number of channels.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSigmoid(Module):
    """Hard Sigmoid: ReLU6(x + 3) / 6.

    Attributes:
        relu6: ReLU6 activation layer.
    """

    def __init__(self, inplace: bool = True) -> None:
        """Initialize HardSigmoid.
        
        Args:
            inplace: Whether to perform the operation in-place. Defaults to True.
        """
        super(HardSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hard sigmoid.
        
        Args:
            x: Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with values in range [0, 1].
        """
        return self.relu6(x + 3.0) / 6.0


class HardSwish(Module):
    """Hard Swish activation function from MobileNetV3.
    
    Approximated swish (SiLU) function using hard sigmoid:
    h-swish(x) = x * h-sigmoid(x) = x * ReLU6(x + 3) / 6
    
    This nonlinearity provides similar benefits to swish but with 
    reduced computational cost, making it suitable for mobile deployment.
    
    Reference: "Searching for MobileNetV3" (Howard et al., ICCV 2019)
    
    Attributes:
        hard_sigmoid: Hard sigmoid activation.
    """

    def __init__(self, inplace: bool = True) -> None:
        """Initialize HardSwish.
        
        Args:
            inplace: Whether to perform the operation in-place. Defaults to True.
        """
        super(HardSwish, self).__init__()
        self.hard_sigmoid = HardSigmoid(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hard swish.
        
        Args:
            x: Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        return x * self.hard_sigmoid(x)


class SqueezeExcite(Module):
    """SE block for MobileNetV3 (cavaface).

    Attributes:
        avg_pool: Global average pooling layer.
        fc: Sequential fully connected layers for SE.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
    ) -> None:
        """Initialize SqueezeExcite block.
        
        Args:
            channels: Number of input/output channels.
            reduction: Reduction ratio for squeeze operation. Defaults to 4.
        """
        super(SqueezeExcite, self).__init__()
        reduced_channels = _make_divisible(channels // reduction, 8)
        
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc = Sequential(
            Linear(channels, reduced_channels),
            ReLU(inplace=True),
            Linear(reduced_channels, channels),
            HardSigmoid(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SE block.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Recalibrated tensor of shape (N, C, H, W).
        """
        batch_size, channels, _, _ = x.size()
        # Squeeze: global average pooling
        y = self.avg_pool(x).view(batch_size, channels)
        # Excite: FC layers with sigmoid
        y = self.fc(y).view(batch_size, channels, 1, 1)
        # Scale: channel-wise multiplication
        return x * y


class ConvBNActivation(Module):
    """Convolution block with Conv-BN-Activation structure for MobileNetV3.
    
    Flexible convolution block supporting different activation functions.
    This implementation follows cavaface's version which uses PReLU
    for better face recognition performance.
    
    Attributes:
        conv: Convolution layer.
        bn: Batch normalization layer.
        activation: Activation layer (or None).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: str = "prelu",
    ) -> None:
        """Initialize ConvBNActivation.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel. Defaults to 3.
            stride: Convolution stride. Defaults to 1.
            groups: Number of groups for grouped convolution. Defaults to 1.
            activation: Activation type ("prelu", "relu", "hswish", or None).
                Defaults to "prelu" (cavaface style).
        """
        super(ConvBNActivation, self).__init__()
        padding = (kernel_size - 1) // 2
        
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
        
        if activation == "prelu":
            self.activation = nn.PReLU(out_channels)
        elif activation == "relu":
            self.activation = ReLU(inplace=True)
        elif activation == "hswish":
            self.activation = HardSwish(inplace=True)
        else:
            self.activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolution block.
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H', W').
        """
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class InvertedResidualV3(Module):
    """Inverted residual block for MobileNetV3 (cavaface).

    Attributes:
        use_residual: Whether to use residual connection.
        conv: Sequential convolution layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        use_hswish: bool,  # Ignored in cavaface version, always uses PReLU
    ) -> None:
        """Initialize InvertedResidualV3 block.
        
        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of expanded (hidden) channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size for depthwise convolution (3 or 5).
            stride: Stride for depthwise convolution (1 or 2).
            use_se: Whether to use Squeeze-and-Excitation.
            use_hswish: Ignored in this implementation (cavaface uses PReLU for all).
        """
        super(InvertedResidualV3, self).__init__()
        assert stride in [1, 2], "Stride must be 1 or 2"
        assert kernel_size in [3, 5], "Kernel size must be 3 or 5"
        
        self.use_residual = (stride == 1) and (in_channels == out_channels)
        padding = (kernel_size - 1) // 2
        
        if in_channels == hidden_channels:
            layers = [
                # Depthwise conv
                Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=hidden_channels,
                    bias=False,
                ),
                BatchNorm2d(hidden_channels),
                nn.PReLU(hidden_channels),
            ]
            # SE block (optional)
            if use_se:
                layers.append(SqueezeExcite(hidden_channels))
            # Pointwise projection (linear)
            layers.extend([
                Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
                BatchNorm2d(out_channels),
            ])
        else:
            layers = [
                # Pointwise expansion
                Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                BatchNorm2d(hidden_channels),
                nn.PReLU(hidden_channels),
                # Depthwise conv
                Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=hidden_channels,
                    bias=False,
                ),
                BatchNorm2d(hidden_channels),
            ]
            # SE block (optional) - placed after DW BN, before PReLU
            if use_se:
                layers.append(SqueezeExcite(hidden_channels))
            # PReLU after SE
            layers.append(nn.PReLU(hidden_channels))
            # Pointwise projection (linear)
            layers.extend([
                Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
                BatchNorm2d(out_channels),
            ])
        
        self.conv = Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the inverted residual block.
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H', W').
        """
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class GDC(Module):
    """Global Depthwise Convolution for face embedding output.
    
    Replaces global average pooling with a depthwise convolution followed
    by a pointwise convolution for generating face embeddings. This approach
    is commonly used in face recognition networks.
    
    Attributes:
        dw_conv: Global depthwise convolution.
        flatten: Flatten layer.
        fc: Fully connected layer for embedding.
        bn: Batch normalization for embedding.
    """

    def __init__(
        self,
        in_channels: int,
        num_features: int,
        input_size: int = 7,
    ) -> None:
        """Initialize GDC.
        
        Args:
            in_channels: Number of input channels.
            num_features: Output embedding dimension.
            input_size: Spatial size of input feature map. Defaults to 7.
        """
        super(GDC, self).__init__()
        self.dw_conv = Conv2d(
            in_channels,
            in_channels,
            kernel_size=input_size,
            stride=1,
            padding=0,
            groups=in_channels,
            bias=False,
        )
        self.bn_dw = BatchNorm2d(in_channels)
        self.fc = Linear(in_channels, num_features, bias=False)
        self.bn = BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GDC.
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Face embeddings of shape (N, num_features).
        """
        x = self.dw_conv(x)
        x = self.bn_dw(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x


MOBILENETV3_LARGE_CFG = [
    # k,  t,    c,  SE, HS, s
    [3,   1,   16,  0,  0,  1],   # Layer 1
    [3,   4,   24,  0,  0,  2],   # Layer 2
    [3,   3,   24,  0,  0,  1],   # Layer 3
    [5,   3,   40,  1,  0,  2],   # Layer 4
    [5,   3,   40,  1,  0,  1],   # Layer 5
    [5,   3,   40,  1,  0,  1],   # Layer 6
    [3,   6,   80,  0,  1,  2],   # Layer 7
    [3, 2.5,   80,  0,  1,  1],   # Layer 8
    [3, 2.3,   80,  0,  1,  1],   # Layer 9
    [3, 2.3,   80,  0,  1,  1],   # Layer 10
    [3,   6,  112,  1,  1,  1],   # Layer 11
    [3,   6,  112,  1,  1,  1],   # Layer 12
    [5,   6,  160,  1,  1,  2],   # Layer 13
    [5,   6,  160,  1,  1,  1],   # Layer 14
    [5,   6,  160,  1,  1,  1],   # Layer 15
]

MOBILENETV3_SMALL_CFG = [
    # k,  t,    c,  SE, HS, s
    [3,    1,   16,  1,  0,  2],   # Layer 1
    [3,  4.5,   24,  0,  0,  2],   # Layer 2
    [3, 3.67,   24,  0,  0,  1],   # Layer 3
    [5,    4,   40,  1,  1,  2],   # Layer 4
    [5,    6,   40,  1,  1,  1],   # Layer 5
    [5,    6,   40,  1,  1,  1],   # Layer 6
    [5,    3,   48,  1,  1,  1],   # Layer 7
    [5,    3,   48,  1,  1,  1],   # Layer 8
    [5,    6,   96,  1,  1,  2],   # Layer 9
    [5,    6,   96,  1,  1,  1],   # Layer 10
    [5,    6,   96,  1,  1,  1],   # Layer 11
]


class MobileNetV3(Module):
    """MobileNetV3 backbone for face recognition (cavaface variant).

    Attributes:
        num_features: Output embedding dimension.
        mode: Model variant ("large" or "small").
        width_mult: Width multiplier for channel scaling.
    """

    def __init__(
        self,
        mode: str = "large",
        width_mult: float = 1.0,
        input_size: List[int] = [112, 112],
        num_features: int = 512,
    ) -> None:
        """Initialize MobileNetV3 backbone.
        
        Args:
            mode: Model variant, either "large" or "small". Large provides
                higher accuracy while Small is more efficient. Defaults to "large".
            width_mult: Width multiplier for scaling channel dimensions.
                Defaults to 1.0.
            input_size: Input image size as [H, W]. Only [112, 112] is supported.
                Defaults to [112, 112].
            num_features: Output embedding dimension. Defaults to 512.
        """
        super(MobileNetV3, self).__init__()
        assert mode in ["large", "small"], "mode must be 'large' or 'small'"
        assert input_size[0] in [112], "input_size should be [112, 112]. Other sizes may cause GDC layer issues."
        
        self.num_features = num_features
        self.mode = mode
        self.width_mult = width_mult
        
        # Select configuration based on mode
        if mode == "large":
            cfg = MOBILENETV3_LARGE_CFG
        else:
            cfg = MOBILENETV3_SMALL_CFG
        
        # cavaface uses fixed 512 channels for last conv
        last_conv_channels = 512
        
        # Build feature extraction layers
        features = []
        
        # First layer: 3x3 conv with stride 1 (adapted for 112x112 input)
        # cavaface uses stride 1 for 112x112 input
        input_channel = _make_divisible(16 * width_mult, 8)
        features.append(
            ConvBNActivation(3, input_channel, kernel_size=3, stride=1, activation="prelu")
        )
        
        # Inverted residual blocks
        for k, t, c, use_se, use_hs, s in cfg:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            features.append(
                InvertedResidualV3(
                    in_channels=input_channel,
                    hidden_channels=exp_size,
                    out_channels=output_channel,
                    kernel_size=k,
                    stride=s,
                    use_se=bool(use_se),
                    use_hswish=bool(use_hs),  # Ignored, uses PReLU
                )
            )
            input_channel = output_channel
        
        # Last conv layer: 1x1 conv to expand to 512 channels (cavaface)
        features.append(
            ConvBNActivation(
                input_channel,
                last_conv_channels,
                kernel_size=1,
                activation="prelu",
            )
        )
        
        self.features = Sequential(*features)
        
        # Calculate feature map size after all conv layers
        feature_map_size = input_size[0] // 16  # 112 -> 7
        
        # Output layer: Global Depthwise Convolution for face embedding
        self.output_layer = GDC(
            in_channels=last_conv_channels,
            num_features=num_features,
            input_size=feature_map_size,
        )
        
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights.
        
        Follows the initialization scheme from the original MobileNetV3:
        - Conv2d: Kaiming normal initialization (fan_out mode)
        - BatchNorm: weight=1, bias=0
        - Linear: Kaiming normal initialization
        """
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (BatchNorm2d, BatchNorm1d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embeddings from input images.
        
        Args:
            x: Input images of shape (N, 3, H, W).
            
        Returns:
            torch.Tensor: Face embeddings of shape (N, num_features).
        """
        x = self.features(x)
        x = self.output_layer(x)
        return x


def MobileNetV3_Large(**kwargs) -> MobileNetV3:
    """Construct a MobileNetV3-Large model.
    
    The large variant of MobileNetV3, providing higher accuracy at the cost
    of more computation. This implementation follows cavaface's version
    optimized for face recognition with PReLU activation.
    
    Note:
        Only supports 112x112 input. Using other input sizes may cause
        issues with the GDC output layer.
    
    Keyword Args:
        width_mult (float): Width multiplier for scaling. Defaults to 1.0.
        input_size (List[int]): Input image size. Only [112, 112] is supported.
            Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        
    Returns:
        MobileNetV3: MobileNetV3-Large model instance.
    """
    return MobileNetV3(mode="large", **kwargs)


def MobileNetV3_Small(**kwargs) -> MobileNetV3:
    """Construct a MobileNetV3-Small model.
    
    The small variant of MobileNetV3, optimized for low-latency applications.
    This implementation follows cavaface's version with PReLU activation
    for better face recognition performance.
    
    Note:
        Only supports 112x112 input. Using other input sizes may cause
        issues with the GDC output layer.
    
    Keyword Args:
        width_mult (float): Width multiplier for scaling. Defaults to 1.0.
        input_size (List[int]): Input image size. Only [112, 112] is supported.
            Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        
    Returns:
        MobileNetV3: MobileNetV3-Small model instance.
    """
    return MobileNetV3(mode="small", **kwargs)


# =============================================================================
# MobileNetV4 Implementation
# =============================================================================


class DropPath(Module):
    """Drop paths (Stochastic Depth) per sample.
    
    When applied in main path of residual blocks, this drops the entire
    residual branch with probability `drop_prob`.
    
    Reference: "Deep Networks with Stochastic Depth" - https://arxiv.org/abs/1603.09382
    
    Attributes:
        drop_prob: Probability of dropping the path.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        """Initialize DropPath.
        
        Args:
            drop_prob: Probability of dropping the path. Defaults to 0.0.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply drop path to input tensor.
        
        Args:
            x: Input tensor.
            
        Returns:
            torch.Tensor: Output tensor with drop path applied.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with arbitrary number of dimensions
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class LayerScale2d(Module):
    """Layer Scale for 2D feature maps (NCHW format).
    
    Applies per-channel learnable scaling to feature maps, which helps
    stabilize training of deeper networks.
    
    Reference: "Going deeper with Image Transformers" - https://arxiv.org/abs/2103.17239
    
    Attributes:
        gamma: Learnable scaling parameter.
    """

    def __init__(
        self,
        dim: int,
        init_value: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        """Initialize LayerScale2d.
        
        Args:
            dim: Number of channels.
            init_value: Initial value for scaling parameters. Defaults to 1e-5.
            inplace: Whether to apply operation inplace. Defaults to False.
        """
        super(LayerScale2d, self).__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer scale to input tensor.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Scaled output tensor.
        """
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


class ConvBNAct(Module):
    """Convolution block with Conv-BN-Activation structure for MobileNetV4.
    
    Flexible convolution block supporting different activation functions
    (ReLU, GELU, or None for linear).
    
    Attributes:
        conv: Convolution layer.
        bn: Batch normalization layer.
        act: Activation layer (or None).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        bias: bool = False,
        activation: str = "relu",
    ) -> None:
        """Initialize ConvBNAct.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel. Defaults to 3.
            stride: Convolution stride. Defaults to 1.
            padding: Convolution padding. If None, uses (kernel_size - 1) // 2.
            groups: Number of groups for grouped convolution. Defaults to 1.
            bias: Whether to use bias. Defaults to False.
            activation: Type of activation ('relu', 'gelu', or None). Defaults to 'relu'.
        """
        super(ConvBNAct, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = BatchNorm2d(out_channels)
        
        if activation == "relu":
            self.act = ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolution block.
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H', W').
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class UniversalInvertedBottleneck(Module):
    """Universal Inverted Bottleneck (UIB) block for MobileNetV4.
    
    A unified and flexible block structure that can represent:
    1. Inverted Bottleneck (IB): dw_kernel_start=0, dw_kernel_mid>0, dw_kernel_end=0
    2. ConvNeXt: dw_kernel_start>0, dw_kernel_mid=0, dw_kernel_end=0
    3. Feed Forward Network (FFN): dw_kernel_start=0, dw_kernel_mid=0, dw_kernel_end=0
    4. Extra Depthwise (ExtraDW): dw_kernel_start>0, dw_kernel_mid>0, dw_kernel_end=0
    
    The block structure:
    - Optional starting depthwise conv (before expansion)
    - 1x1 expansion conv
    - Optional middle depthwise conv (standard IB position)
    - 1x1 projection conv (linear, no activation)
    - Optional ending depthwise conv (after projection)
    - Residual connection if stride=1 and in_channels=out_channels
    
    Reference: "MobileNetV4 -- Universal Models for the Mobile Ecosystem"
    (Qin et al., arXiv 2024)
    
    Attributes:
        has_skip: Whether to use residual connection.
        dw_start: Starting depthwise convolution (or Identity).
        pw_exp: Pointwise expansion convolution.
        dw_mid: Middle depthwise convolution (or Identity).
        pw_proj: Pointwise projection convolution.
        dw_end: Ending depthwise convolution (or Identity).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4.0,
        dw_kernel_start: int = 0,
        dw_kernel_mid: int = 3,
        dw_kernel_end: int = 0,
        stride: int = 1,
        activation: str = "relu",
        drop_path_rate: float = 0.0,
        layer_scale_init_value: Optional[float] = None,
    ) -> None:
        """Initialize UniversalInvertedBottleneck.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            expand_ratio: Expansion ratio for hidden channels. Defaults to 4.0.
            dw_kernel_start: Kernel size for starting depthwise conv (0 to disable).
            dw_kernel_mid: Kernel size for middle depthwise conv (0 to disable).
            dw_kernel_end: Kernel size for ending depthwise conv (0 to disable).
            stride: Stride for the block (applied to first available dw conv).
            activation: Activation function ('relu' or 'gelu'). Defaults to 'relu'.
            drop_path_rate: Drop path rate for stochastic depth. Defaults to 0.0.
            layer_scale_init_value: Initial value for layer scale. None to disable.
        """
        super(UniversalInvertedBottleneck, self).__init__()
        
        self.has_skip = (in_channels == out_channels) and (stride == 1)
        mid_channels = _make_divisible(in_channels * expand_ratio)
        
        # Stride placement logic (following timm):
        # - If dw_mid exists (dw_kernel_mid > 0), stride goes to dw_mid
        # - Else if dw_start exists (dw_kernel_start > 0), stride goes to dw_start
        # - Else if dw_end exists (dw_kernel_end > 0), stride goes to dw_end
        
        # Starting depthwise convolution (before expansion)
        if dw_kernel_start > 0:
            # Apply stride here only if there's no dw_mid
            dw_start_stride = stride if dw_kernel_mid == 0 else 1
            self.dw_start = ConvBNAct(
                in_channels,
                in_channels,
                kernel_size=dw_kernel_start,
                stride=dw_start_stride,
                groups=in_channels,
                activation=None,  # No activation after dw_start
            )
        else:
            self.dw_start = nn.Identity()
        
        # Pointwise expansion
        self.pw_exp = ConvBNAct(
            in_channels,
            mid_channels,
            kernel_size=1,
            activation=activation,
        )
        
        # Middle depthwise convolution
        if dw_kernel_mid > 0:
            # Always apply stride to dw_mid if it exists
            self.dw_mid = ConvBNAct(
                mid_channels,
                mid_channels,
                kernel_size=dw_kernel_mid,
                stride=stride,
                groups=mid_channels,
                activation=activation,
            )
        else:
            self.dw_mid = nn.Identity()
        
        # Pointwise projection (linear, no activation)
        self.pw_proj = ConvBNAct(
            mid_channels,
            out_channels,
            kernel_size=1,
            activation=None,
        )
        
        # Ending depthwise convolution (after projection)
        if dw_kernel_end > 0:
            # Apply stride here only if neither dw_start nor dw_mid exist
            dw_end_stride = stride if (dw_kernel_start == 0 and dw_kernel_mid == 0) else 1
            self.dw_end = ConvBNAct(
                out_channels,
                out_channels,
                kernel_size=dw_kernel_end,
                stride=dw_end_stride,
                groups=out_channels,
                activation=None,
            )
        else:
            self.dw_end = nn.Identity()
        
        # Layer scale for stabilizing training (following timm)
        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(out_channels, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()
        
        # Drop path for stochastic depth regularization
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the UIB block.
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H', W').
        """
        shortcut = x
        
        x = self.dw_start(x)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.pw_proj(x)
        x = self.dw_end(x)
        x = self.layer_scale(x)
        
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        
        return x


class MultiQueryAttention(Module):
    """Mobile Multi-Query Attention (MQA) block for MobileNetV4 Hybrid.
    
    Attributes:
        num_heads: Number of attention heads.
        key_dim: Dimension of each key/query head.
        value_dim: Dimension of each value head.
        scale: Scaling factor for attention scores.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 8,
        key_dim: int = 64,
        value_dim: int = 64,
        kv_stride: int = 1,
        dw_kernel_size: int = 3,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: Optional[float] = None,
    ) -> None:
        """Initialize MultiQueryAttention.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_heads: Number of attention heads. Defaults to 8.
            key_dim: Dimension of each key/query head. Defaults to 64.
            value_dim: Dimension of each value head. Defaults to 64.
            kv_stride: Stride for key-value spatial downsampling. Defaults to 1.
            dw_kernel_size: Kernel size for depthwise conv in KV downsampling.
            drop_path_rate: Drop path rate for stochastic depth. Defaults to 0.0.
            layer_scale_init_value: Initial value for layer scale. None to disable.
        """
        super(MultiQueryAttention, self).__init__()
        
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.kv_stride = kv_stride
        self.scale = key_dim ** -0.5
        
        self.has_skip = (in_channels == out_channels)
        
        # Normalization before attention
        self.norm = BatchNorm2d(in_channels)
        
        # Query projection: per-head
        self.query = Conv2d(in_channels, num_heads * key_dim, kernel_size=1, bias=False)
        
        # Key and Value projections: shared across heads (multi-query)
        if kv_stride > 1:
            self.key = Sequential(
                Conv2d(in_channels, in_channels, kernel_size=dw_kernel_size, 
                       stride=kv_stride, padding=dw_kernel_size // 2, 
                       groups=in_channels, bias=False),
                BatchNorm2d(in_channels),
                Conv2d(in_channels, key_dim, kernel_size=1, bias=False),
            )
            self.value = Sequential(
                Conv2d(in_channels, in_channels, kernel_size=dw_kernel_size,
                       stride=kv_stride, padding=dw_kernel_size // 2,
                       groups=in_channels, bias=False),
                BatchNorm2d(in_channels),
                Conv2d(in_channels, value_dim, kernel_size=1, bias=False),
            )
        else:
            self.key = Conv2d(in_channels, key_dim, kernel_size=1, bias=False)
            self.value = Conv2d(in_channels, value_dim, kernel_size=1, bias=False)
        
        # Output projection
        self.output = Sequential(
            Conv2d(num_heads * value_dim, out_channels, kernel_size=1, bias=False),
            BatchNorm2d(out_channels),
        )
        
        # Layer scale for stabilizing training (following timm)
        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(out_channels, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()
        
        # Drop path for stochastic depth regularization
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        
        # Initialize weights using xavier_uniform (following timm)
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize attention weights using xavier_uniform for stability."""
        nn.init.xavier_uniform_(self.query.weight)
        if isinstance(self.key, Conv2d):
            nn.init.xavier_uniform_(self.key.weight)
        else:
            # Sequential with multiple layers
            for m in self.key.modules():
                if isinstance(m, Conv2d):
                    nn.init.xavier_uniform_(m.weight)
        if isinstance(self.value, Conv2d):
            nn.init.xavier_uniform_(self.value.weight)
        else:
            for m in self.value.modules():
                if isinstance(m, Conv2d):
                    nn.init.xavier_uniform_(m.weight)
        for m in self.output.modules():
            if isinstance(m, Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MQA block.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W).
        """
        shortcut = x
        B, C, H, W = x.shape
        
        x = self.norm(x)
        
        # Query: [B, num_heads * key_dim, H, W] -> [B, num_heads, H*W, key_dim]
        q = self.query(x)
        q = q.reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        
        # Key: [B, key_dim, H', W'] -> [B, 1, H'*W', key_dim]
        k = self.key(x)
        Hk, Wk = k.shape[2], k.shape[3]
        k = k.reshape(B, self.key_dim, Hk * Wk).permute(0, 2, 1).unsqueeze(1)
        
        # Value: [B, value_dim, H', W'] -> [B, 1, H'*W', value_dim]
        v = self.value(x)
        v = v.reshape(B, self.value_dim, Hk * Wk).permute(0, 2, 1).unsqueeze(1)
        
        # Attention: [B, num_heads, H*W, key_dim] @ [B, 1, key_dim, H'*W'] 
        #          -> [B, num_heads, H*W, H'*W']
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Output: [B, num_heads, H*W, H'*W'] @ [B, 1, H'*W', value_dim]
        #       -> [B, num_heads, H*W, value_dim]
        out = attn @ v
        
        # Reshape: [B, num_heads, H*W, value_dim] -> [B, num_heads * value_dim, H, W]
        out = out.permute(0, 1, 3, 2).reshape(B, self.num_heads * self.value_dim, H, W)
        
        out = self.output(out)
        out = self.layer_scale(out)
        
        if self.has_skip:
            out = self.drop_path(out) + shortcut
        
        return out


class EdgeResidual(Module):
    """Edge Residual (Fused Inverted Bottleneck) block for MobileNetV4.
    
    This is a fused version of the inverted bottleneck where the first
    pointwise expansion and depthwise convolution are combined into a 
    single 3x3 convolution, followed by a 1x1 projection.
    
    Structure: 3x3 Conv (expansion) -> 1x1 Conv (projection)
    
    This block is used in the early stages of MobileNetV4 where fused
    convolutions are more efficient than depthwise separable convolutions.
    
    Attributes:
        has_skip: Whether to use residual connection.
        conv_exp: Expansion convolution (3x3).
        conv_proj: Projection convolution (1x1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4.0,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "relu",
        drop_path_rate: float = 0.0,
    ) -> None:
        """Initialize EdgeResidual.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            expand_ratio: Expansion ratio for hidden channels. Defaults to 4.0.
            kernel_size: Kernel size for expansion conv. Defaults to 3.
            stride: Stride for the block. Defaults to 1.
            activation: Activation function. Defaults to 'relu'.
            drop_path_rate: Drop path rate for stochastic depth. Defaults to 0.0.
        """
        super(EdgeResidual, self).__init__()
        
        self.has_skip = (in_channels == out_channels) and (stride == 1)
        mid_channels = _make_divisible(in_channels * expand_ratio)
        
        # Fused expansion: 3x3 conv
        self.conv_exp = ConvBNAct(
            in_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
        )
        
        # Projection: 1x1 conv (linear)
        self.conv_proj = ConvBNAct(
            mid_channels,
            out_channels,
            kernel_size=1,
            activation=None,
        )
        
        # Drop path for stochastic depth regularization
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the EdgeResidual block.
        
        Args:
            x: Input tensor of shape (N, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H', W').
        """
        shortcut = x
        x = self.conv_exp(x)
        x = self.conv_proj(x)
        
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        
        return x


# =============================================================================
# MobileNetV4 Architecture Configurations
# =============================================================================


MOBILENETV4_CONV_SMALL_CFG = [
    # Stage 0: stem output
    # Stage 1, 56x56 in
    [
        ('cn', 32, 2, 0, 0, 0),       # Conv stem: 3x3 conv, stride 2
        ('er', 32, 1, 1, 0, 0),       # FusedIB
    ],
    # Stage 2, 28x28 in
    [
        ('er', 64, 2, 4, 0, 0),       # FusedIB
        ('cn', 64, 1, 1, 0, 0),       # Conv
    ],
    # Stage 3, 14x14 in
    [
        ('uib', 96, 2, 3, 5, 5),      # ExtraDW
        ('uib', 96, 1, 2, 0, 3),      # IR
        ('uib', 96, 1, 2, 0, 3),      # IR
        ('uib', 96, 1, 2, 0, 3),      # IR
        ('uib', 96, 1, 2, 0, 3),      # IR
        ('uib', 96, 1, 4, 3, 0),      # ConvNeXt
    ],
    # Stage 4, 7x7 in
    [
        ('uib', 128, 2, 6, 3, 3),     # ExtraDW
        ('uib', 128, 1, 4, 0, 5),     # IR
        ('uib', 128, 1, 3, 0, 5),     # IR
        ('uib', 128, 1, 4, 0, 3),     # IR
        ('uib', 128, 1, 4, 0, 3),     # IR
    ],
    # Stage 5: head
    [
        ('cn', 960, 1, 0, 0, 0),      # Conv: 1x1 expansion
    ],
]

MOBILENETV4_CONV_MEDIUM_CFG = [
    # Stage 0: stem
    # Stage 1, 56x56 in
    [
        ('cn', 32, 2, 0, 0, 0),       # Conv stem
        ('er', 48, 2, 4, 0, 0),       # FusedIB
    ],
    # Stage 2, 28x28 in
    [
        ('uib', 80, 2, 4, 3, 5),      # ExtraDW
        ('uib', 80, 1, 2, 3, 3),      # ExtraDW
    ],
    # Stage 3, 14x14 in
    [
        ('uib', 160, 2, 6, 3, 5),     # ExtraDW
        ('uib', 160, 1, 2, 0, 0),     # FFN
        ('uib', 160, 1, 4, 3, 3),     # ExtraDW
        ('uib', 160, 1, 4, 3, 5),     # ExtraDW
        ('uib', 160, 1, 4, 3, 3),     # ExtraDW
        ('uib', 160, 1, 4, 3, 0),     # ConvNeXt
        ('uib', 160, 1, 2, 0, 0),     # FFN
        ('uib', 160, 1, 4, 3, 0),     # ConvNeXt
    ],
    # Stage 4, 7x7 in
    [
        ('uib', 256, 2, 6, 5, 5),     # ExtraDW
        ('uib', 256, 1, 4, 5, 5),     # ExtraDW
        ('uib', 256, 1, 4, 3, 5),     # ExtraDW
        ('uib', 256, 1, 4, 3, 5),     # ExtraDW
        ('uib', 256, 1, 2, 0, 0),     # FFN
        ('uib', 256, 1, 2, 3, 5),     # ExtraDW
        ('uib', 256, 1, 4, 5, 5),     # ExtraDW
        ('uib', 256, 1, 4, 0, 0),     # FFN
        ('uib', 256, 1, 4, 0, 0),     # FFN
        ('uib', 256, 1, 4, 3, 0),     # ConvNeXt
        ('uib', 256, 1, 4, 5, 0),     # ConvNeXt
    ],
    # Stage 5: head
    [
        ('cn', 960, 1, 0, 0, 0),      # Conv
    ],
]

MOBILENETV4_CONV_LARGE_CFG = [
    # Stage 0: stem
    # Stage 1, 56x56 in
    [
        ('cn', 24, 2, 0, 0, 0),       # Conv stem
        ('er', 48, 2, 4, 0, 0),       # FusedIB
    ],
    # Stage 2, 28x28 in
    [
        ('uib', 96, 2, 4, 3, 5),      # ExtraDW
        ('uib', 96, 1, 4, 3, 3),      # ExtraDW
    ],
    # Stage 3, 14x14 in
    [
        ('uib', 192, 2, 4, 3, 5),     # ExtraDW
        ('uib', 192, 1, 4, 3, 3),     # ExtraDW
        ('uib', 192, 1, 4, 3, 3),     # ExtraDW
        ('uib', 192, 1, 4, 3, 3),     # ExtraDW
        ('uib', 192, 1, 4, 3, 5),     # ExtraDW
        ('uib', 192, 1, 4, 5, 3),     # ExtraDW
        ('uib', 192, 1, 4, 5, 3),     # ExtraDW
        ('uib', 192, 1, 4, 5, 3),     # ExtraDW
        ('uib', 192, 1, 4, 5, 3),     # ExtraDW
        ('uib', 192, 1, 4, 5, 3),     # ExtraDW
        ('uib', 192, 1, 4, 3, 0),     # ConvNeXt
    ],
    # Stage 4, 7x7 in
    [
        ('uib', 512, 2, 4, 5, 5),     # ExtraDW
        ('uib', 512, 1, 4, 5, 5),     # ExtraDW
        ('uib', 512, 1, 4, 5, 5),     # ExtraDW
        ('uib', 512, 1, 4, 5, 5),     # ExtraDW
        ('uib', 512, 1, 4, 5, 0),     # ConvNeXt
        ('uib', 512, 1, 4, 5, 3),     # ExtraDW
        ('uib', 512, 1, 4, 5, 0),     # ConvNeXt
        ('uib', 512, 1, 4, 5, 0),     # ConvNeXt
        ('uib', 512, 1, 4, 5, 3),     # ExtraDW
        ('uib', 512, 1, 4, 5, 5),     # ExtraDW
        ('uib', 512, 1, 4, 5, 3),     # ExtraDW
        ('uib', 512, 1, 4, 5, 0),     # ConvNeXt
    ],
    # Stage 5: head
    [
        ('cn', 960, 1, 0, 0, 0),      # Conv
    ],
]

# MobileNetV4-Hybrid-Medium configuration (with Mobile MQA)
MOBILENETV4_HYBRID_MEDIUM_CFG = [
    # Stage 0: stem
    # Stage 1, 56x56 in
    [
        ('cn', 32, 2, 0, 0, 0),       # Conv stem
        ('er', 48, 2, 4, 0, 0),       # FusedIB
    ],
    # Stage 2, 28x28 in
    [
        ('uib', 80, 2, 4, 3, 5),      # ExtraDW
        ('uib', 80, 1, 2, 3, 3),      # ExtraDW
    ],
    # Stage 3, 14x14 in
    [
        ('uib', 160, 2, 6, 3, 5),     # ExtraDW
        ('uib', 160, 1, 4, 3, 3),     # ExtraDW
        ('uib', 160, 1, 4, 3, 5),     # ExtraDW
        ('uib', 160, 1, 4, 3, 3),     # ExtraDW
        ('mqa', 160, 1, 0, 0, 0, 4, 64, 64, 2),  # MQA with KV downsample
        ('uib', 160, 1, 4, 3, 3),     # ExtraDW
        ('mqa', 160, 1, 0, 0, 0, 4, 64, 64, 2),  # MQA with KV downsample
        ('uib', 160, 1, 4, 3, 0),     # ConvNeXt
        ('mqa', 160, 1, 0, 0, 0, 4, 64, 64, 2),  # MQA with KV downsample
        ('uib', 160, 1, 4, 3, 3),     # ExtraDW
        ('mqa', 160, 1, 0, 0, 0, 4, 64, 64, 2),  # MQA with KV downsample
        ('uib', 160, 1, 4, 3, 0),     # ConvNeXt
    ],
    # Stage 4, 7x7 in
    [
        ('uib', 256, 2, 6, 5, 5),     # ExtraDW
        ('uib', 256, 1, 4, 5, 5),     # ExtraDW
        ('uib', 256, 1, 4, 3, 5),     # ExtraDW
        ('uib', 256, 1, 4, 3, 5),     # ExtraDW
        ('uib', 256, 1, 2, 0, 0),     # FFN
        ('uib', 256, 1, 4, 0, 0),     # FFN
        ('mqa', 256, 1, 0, 0, 0, 4, 64, 64, 1),  # MQA
        ('uib', 256, 1, 4, 3, 0),     # ConvNeXt
        ('mqa', 256, 1, 0, 0, 0, 4, 64, 64, 1),  # MQA
        ('uib', 256, 1, 4, 5, 5),     # ExtraDW
        ('mqa', 256, 1, 0, 0, 0, 4, 64, 64, 1),  # MQA
        ('uib', 256, 1, 4, 5, 0),     # ConvNeXt
        ('mqa', 256, 1, 0, 0, 0, 4, 64, 64, 1),  # MQA
        ('uib', 256, 1, 4, 5, 0),     # ConvNeXt
    ],
    # Stage 5: head
    [
        ('cn', 960, 1, 0, 0, 0),      # Conv
    ],
]

MOBILENETV4_HYBRID_LARGE_CFG = [
    # Stage 0: stem
    # Stage 1, 56x56 in
    [
        ('cn', 24, 2, 0, 0, 0),       # Conv stem
        ('er', 48, 2, 4, 0, 0),       # FusedIB
    ],
    # Stage 2, 28x28 in
    [
        ('uib', 96, 2, 4, 3, 5),      # ExtraDW
        ('uib', 96, 1, 4, 3, 3),      # ExtraDW
    ],
    # Stage 3, 14x14 in
    [
        ('uib', 192, 2, 4, 3, 5),     # ExtraDW
        ('uib', 192, 1, 4, 3, 3),     # ExtraDW
        ('uib', 192, 1, 4, 3, 3),     # ExtraDW
        ('uib', 192, 1, 4, 3, 3),     # ExtraDW
        ('uib', 192, 1, 4, 3, 5),     # ExtraDW
        ('uib', 192, 1, 4, 5, 3),     # ExtraDW
        ('mqa', 192, 1, 0, 0, 0, 8, 48, 48, 2),  # MQA with KV downsample
        ('uib', 192, 1, 4, 5, 3),     # ExtraDW
        ('mqa', 192, 1, 0, 0, 0, 8, 48, 48, 2),  # MQA with KV downsample
        ('uib', 192, 1, 4, 5, 3),     # ExtraDW
        ('mqa', 192, 1, 0, 0, 0, 8, 48, 48, 2),  # MQA with KV downsample
        ('uib', 192, 1, 4, 5, 3),     # ExtraDW
        ('mqa', 192, 1, 0, 0, 0, 8, 48, 48, 2),  # MQA with KV downsample
        ('uib', 192, 1, 4, 3, 0),     # ConvNeXt
    ],
    # Stage 4, 7x7 in
    [
        ('uib', 512, 2, 4, 5, 5),     # ExtraDW
        ('uib', 512, 1, 4, 5, 5),     # ExtraDW
        ('uib', 512, 1, 4, 5, 5),     # ExtraDW
        ('uib', 512, 1, 4, 5, 5),     # ExtraDW
        ('uib', 512, 1, 4, 5, 0),     # ConvNeXt
        ('uib', 512, 1, 4, 5, 3),     # ExtraDW
        ('uib', 512, 1, 4, 5, 0),     # ConvNeXt
        ('uib', 512, 1, 4, 5, 0),     # ConvNeXt
        ('mqa', 512, 1, 0, 0, 0, 8, 64, 64, 1),  # MQA
        ('uib', 512, 1, 4, 5, 0),     # ConvNeXt
        ('mqa', 512, 1, 0, 0, 0, 8, 64, 64, 1),  # MQA
        ('uib', 512, 1, 4, 5, 0),     # ConvNeXt
        ('mqa', 512, 1, 0, 0, 0, 8, 64, 64, 1),  # MQA
        ('uib', 512, 1, 4, 5, 0),     # ConvNeXt
        ('mqa', 512, 1, 0, 0, 0, 8, 64, 64, 1),  # MQA
        ('uib', 512, 1, 4, 5, 0),     # ConvNeXt
    ],
    # Stage 5: head
    [
        ('cn', 960, 1, 0, 0, 0),      # Conv
    ],
]


class MobileNetV4(Module):
    """MobileNetV4 backbone for face recognition.

    Attributes:
        num_features: Output embedding dimension.
        variant: Model variant name.
    """

    def __init__(
        self,
        variant: str = "conv_medium",
        width_mult: float = 1.0,
        input_size: List[int] = [112, 112],
        num_features: int = 512,
        activation: str = "relu",
        drop_path_rate: float = 0.0,
    ) -> None:
        """Initialize MobileNetV4 backbone.
        
        Args:
            variant: Model variant. Options: 'conv_small', 'conv_medium', 
                'conv_large', 'hybrid_medium', 'hybrid_large'. Defaults to 'conv_medium'.
            width_mult: Width multiplier for scaling channel dimensions.
                Defaults to 1.0.
            input_size: Input image size as [H, W]. Currently supports [112, 112].
                Defaults to [112, 112].
            num_features: Output embedding dimension. Defaults to 512.
            activation: Activation function ('relu' or 'gelu'). For hybrid variants,
                'gelu' is recommended. Defaults to 'relu'.
            drop_path_rate: Drop path rate for stochastic depth. Applied linearly
                increasing across blocks. Defaults to 0.0.
        """
        super(MobileNetV4, self).__init__()
        assert variant in ["conv_small", "conv_medium", "conv_large", 
                          "hybrid_medium", "hybrid_large"], \
            f"Invalid variant: {variant}"
        assert input_size[0] in [112], "input_size should be [112, 112]"
        
        self.num_features = num_features
        self.variant = variant
        self.width_mult = width_mult
        
        # Select configuration based on variant
        layer_scale_init_value = None
        if variant == "conv_small":
            cfg = MOBILENETV4_CONV_SMALL_CFG
            activation = "relu"
        elif variant == "conv_medium":
            cfg = MOBILENETV4_CONV_MEDIUM_CFG
            activation = "relu"
        elif variant == "conv_large":
            cfg = MOBILENETV4_CONV_LARGE_CFG
            activation = "relu"
        elif variant == "hybrid_medium":
            cfg = MOBILENETV4_HYBRID_MEDIUM_CFG
            activation = "gelu" if activation == "relu" else activation
            layer_scale_init_value = 1e-5
        else:  # hybrid_large
            cfg = MOBILENETV4_HYBRID_LARGE_CFG
            activation = "gelu" if activation == "relu" else activation
            layer_scale_init_value = 1e-5

        total_downsamples = sum(
            1 for stage_cfg in cfg for block_cfg in stage_cfg if block_cfg[2] > 1
        )
        stem_stride_one = (input_size[0] == 112 and total_downsamples == 5)
        
        # Count total blocks for drop path rate calculation
        total_blocks = sum(
            1 for stage_cfg in cfg for block_cfg in stage_cfg 
            if block_cfg[0] in ('er', 'uib', 'mqa')
        )
        block_idx = 0
        
        # Build feature extraction layers
        features = []
        in_channels = 3
        applied_strides = []
        
        for stage_cfg in cfg:
            for block_cfg in stage_cfg:
                block_type = block_cfg[0]
                out_channels = _make_divisible(block_cfg[1] * width_mult)
                stride = block_cfg[2]
                if block_type == 'cn' and in_channels == 3 and stem_stride_one:
                    stride = 1
                applied_strides.append(stride)
                
                if block_type == 'cn':
                    # Standard convolution block
                    if in_channels == 3:
                        # Stem: 3x3 conv
                        features.append(
                            ConvBNAct(in_channels, out_channels, kernel_size=3, 
                                     stride=stride, activation=activation)
                        )
                    else:
                        # 1x1 expansion conv
                        features.append(
                            ConvBNAct(in_channels, out_channels, kernel_size=1, 
                                     stride=1, activation=activation)
                        )
                    in_channels = out_channels
                    
                elif block_type == 'er':
                    # Edge Residual (Fused Inverted Bottleneck)
                    expand_ratio = block_cfg[3]
                    dpr = drop_path_rate * block_idx / max(total_blocks - 1, 1) if total_blocks > 1 else 0.0
                    features.append(
                        EdgeResidual(in_channels, out_channels, 
                                    expand_ratio=expand_ratio, stride=stride,
                                    activation=activation,
                                    drop_path_rate=dpr)
                    )
                    in_channels = out_channels
                    block_idx += 1
                    
                elif block_type == 'uib':
                    # Universal Inverted Bottleneck
                    expand_ratio = block_cfg[3]
                    dw_kernel_start = block_cfg[4]
                    dw_kernel_mid = block_cfg[5]
                    dpr = drop_path_rate * block_idx / max(total_blocks - 1, 1) if total_blocks > 1 else 0.0
                    features.append(
                        UniversalInvertedBottleneck(
                            in_channels, out_channels,
                            expand_ratio=expand_ratio,
                            dw_kernel_start=dw_kernel_start,
                            dw_kernel_mid=dw_kernel_mid,
                            stride=stride,
                            activation=activation,
                            drop_path_rate=dpr,
                            layer_scale_init_value=layer_scale_init_value,
                        )
                    )
                    in_channels = out_channels
                    block_idx += 1
                    
                elif block_type == 'mqa':
                    # Multi-Query Attention
                    num_heads = block_cfg[6]
                    key_dim = block_cfg[7]
                    value_dim = block_cfg[8]
                    kv_stride = block_cfg[9]
                    dpr = drop_path_rate * block_idx / max(total_blocks - 1, 1) if total_blocks > 1 else 0.0
                    features.append(
                        MultiQueryAttention(
                            in_channels, out_channels,
                            num_heads=num_heads,
                            key_dim=key_dim,
                            value_dim=value_dim,
                            kv_stride=kv_stride,
                            drop_path_rate=dpr,
                            layer_scale_init_value=layer_scale_init_value,
                        )
                    )
                    in_channels = out_channels
                    block_idx += 1
        
        self.features = Sequential(*features)
        
        # Get final channel count
        last_channels = _make_divisible(960 * width_mult)
        
        # Output layer: Global Depthwise Convolution (GDC)
        # Matches MobileNetV2/V3 output design for face recognition.
        #
        # Compute feature map size from the effective strides. At 112x112 input,
        # 5-downsample MobileNetV4 variants neutralize the stem stride so their
        # stages run at 56/28/14/7 and use a 7x7 GDC head.
        feature_map_size = input_size[0]
        for stride in applied_strides:
            if stride > 1:
                feature_map_size = (feature_map_size + stride - 1) // stride
        
        self.output_layer = GDC(
            in_channels=last_channels,
            num_features=num_features,
            input_size=feature_map_size,
        )
        
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights.
        
        Follows the initialization scheme from the original MobileNetV4:
        - Conv2d: Kaiming normal initialization (fan_out mode)
        - BatchNorm: weight=1, bias=0
        - Linear: Kaiming normal initialization
        """
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (BatchNorm2d, BatchNorm1d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embeddings from input images.
        
        Args:
            x: Input images of shape (N, 3, H, W).
            
        Returns:
            torch.Tensor: Face embeddings of shape (N, num_features).
        """
        x = self.features(x)
        x = self.output_layer(x)
        return x


def MobileNetV4_Conv_Small(**kwargs) -> MobileNetV4:
    """Construct a MobileNetV4-Conv-Small model.
    
    The smallest pure convolutional variant of MobileNetV4, designed for 
    maximum efficiency on resource-constrained devices.
    
    Keyword Args:
        width_mult (float): Width multiplier for scaling. Defaults to 1.0.
        input_size (List[int]): Input image size as [H, W]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        
    Returns:
        MobileNetV4: MobileNetV4-Conv-Small model instance.
    """
    return MobileNetV4(variant="conv_small", **kwargs)


def MobileNetV4_Conv_Medium(**kwargs) -> MobileNetV4:
    """Construct a MobileNetV4-Conv-Medium model.
    
    Medium-sized pure convolutional variant of MobileNetV4, offering a 
    balance between accuracy and computational efficiency.
    
    Keyword Args:
        width_mult (float): Width multiplier for scaling. Defaults to 1.0.
        input_size (List[int]): Input image size as [H, W]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        
    Returns:
        MobileNetV4: MobileNetV4-Conv-Medium model instance.
    """
    return MobileNetV4(variant="conv_medium", **kwargs)


def MobileNetV4_Conv_Large(**kwargs) -> MobileNetV4:
    """Construct a MobileNetV4-Conv-Large model.
    
    Large pure convolutional variant of MobileNetV4, providing higher 
    accuracy for scenarios where computational resources are less constrained.
    
    Keyword Args:
        width_mult (float): Width multiplier for scaling. Defaults to 1.0.
        input_size (List[int]): Input image size as [H, W]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        
    Returns:
        MobileNetV4: MobileNetV4-Conv-Large model instance.
    """
    return MobileNetV4(variant="conv_large", **kwargs)


def MobileNetV4_Hybrid_Medium(**kwargs) -> MobileNetV4:
    """Construct a MobileNetV4-Hybrid-Medium model.
    
    Medium-sized hybrid variant of MobileNetV4 that combines convolutional 
    blocks with Mobile Multi-Query Attention (MQA) for enhanced representation
    learning while maintaining mobile-friendly efficiency.
    
    Keyword Args:
        width_mult (float): Width multiplier for scaling. Defaults to 1.0.
        input_size (List[int]): Input image size as [H, W]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        
    Returns:
        MobileNetV4: MobileNetV4-Hybrid-Medium model instance.
    """
    return MobileNetV4(variant="hybrid_medium", **kwargs)


def MobileNetV4_Hybrid_Large(**kwargs) -> MobileNetV4:
    """Construct a MobileNetV4-Hybrid-Large model.
    
    Large hybrid variant of MobileNetV4 with extensive use of Mobile MQA,
    achieving state-of-the-art accuracy for mobile-efficient models.
    This variant achieves 87% ImageNet-1K accuracy with just 3.8ms latency
    on Pixel 8 EdgeTPU.
    
    Keyword Args:
        width_mult (float): Width multiplier for scaling. Defaults to 1.0.
        input_size (List[int]): Input image size as [H, W]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        
    Returns:
        MobileNetV4: MobileNetV4-Hybrid-Large model instance.
    """
    return MobileNetV4(variant="hybrid_large", **kwargs)