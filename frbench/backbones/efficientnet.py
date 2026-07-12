"""
EfficientNetV1 Backbone Model for Face Recognition

Reference Papers: 
    - EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (https://arxiv.org/abs/1905.11946)
    - SE-Net: Squeeze-and-Excitation Networks (https://arxiv.org/abs/1709.01507)
Reference Code: https://github.com/JDAI-CV/FaceX-Zoo/blob/main/backbone/EfficientNets.py

Supported Variants: EfficientNet-B0 to B7
"""
from typing import List, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm1d,
    BatchNorm2d,
    Dropout,
    Linear,
    Sigmoid,
    AdaptiveAvgPool2d,
)

from .utils import Flatten


__all__ = [
    "EfficientNetV1",
    "EfficientNetV1_B0",
    "EfficientNetV1_B1",
    "EfficientNetV1_B2",
    "EfficientNetV1_B3",
    "EfficientNetV1_B4",
    "EfficientNetV1_B5",
    "EfficientNetV1_B6",
    "EfficientNetV1_B7",
]


class Swish(Module):
    """Standard Swish activation: x * sigmoid(x)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwishImplementation(torch.autograd.Function):
    """Memory-efficient Swish implementation using custom autograd."""
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(Module):
    """Memory-efficient Swish activation for training."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SwishImplementation.apply(x)


def round_channels(channels: int, divisor: int = 8) -> int:
    """Round number of channels to be divisible by divisor.
    
    Make the number of channels divisible by 8 for better hardware efficiency.
    Following the official TensorFlow implementation.
    
    Args:
        channels: Original number of channels.
        divisor: Divisibility factor. Defaults to 8.
        
    Returns:
        int: Rounded number of channels.
    """
    new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:  # prevent rounding by more than 10%
        new_channels += divisor
    return int(new_channels)


def drop_connect(inputs: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Drop connect implementation for stochastic depth.
    
    Args:
        inputs: Input tensor of shape (N, C, H, W).
        p: Probability of drop connection (0.0~1.0).
        training: Whether in training mode.
        
    Returns:
        torch.Tensor: Output after drop connection.
    """
    assert 0 <= p <= 1, 'p must be in range of [0,1]'
    
    if not training:
        return inputs
    
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    
    # Generate binary tensor mask according to probability
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    
    output = inputs / keep_prob * binary_tensor
    return output


def calc_tf_padding(
    x: torch.Tensor, 
    kernel_size: int, 
    stride: int = 1, 
    dilation: int = 1
) -> Tuple[int, int, int, int]:
    """Calculate TensorFlow-style 'SAME' padding.
    
    Args:
        x: Input tensor.
        kernel_size: Convolution kernel size.
        stride: Convolution stride. Defaults to 1.
        dilation: Convolution dilation. Defaults to 1.
        
    Returns:
        Tuple[int, int, int, int]: Padding values (left, right, top, bottom).
    """
    height, width = x.size()[2:]
    oh = math.ceil(height / stride)
    ow = math.ceil(width / stride)
    pad_h = max((oh - 1) * stride + (kernel_size - 1) * dilation + 1 - height, 0)
    pad_w = max((ow - 1) * stride + (kernel_size - 1) * dilation + 1 - width, 0)
    return pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2


class ConvBNSwish(Module):
    """Convolution block with BatchNorm and Swish activation.
    
    Attributes:
        conv: Convolution layer.
        bn: Batch normalization layer.
        swish: Swish activation function.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bn_mom: float = 0.01,
        bn_eps: float = 1e-3,
        has_activation: bool = True,
    ) -> None:
        """Initialize ConvBNSwish block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride. Defaults to 1.
            padding: Convolution padding. Defaults to 0.
            groups: Number of groups for grouped convolution. Defaults to 1.
            bn_mom: BatchNorm momentum (PyTorch convention). Defaults to 0.01.
            bn_eps: BatchNorm epsilon. Defaults to 1e-3.
            has_activation: Whether to use activation. Defaults to True.
        """
        super(ConvBNSwish, self).__init__()
        self.conv = Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=False
        )
        self.bn = BatchNorm2d(out_channels, momentum=bn_mom, eps=bn_eps)
        self.swish = MemoryEfficientSwish() if has_activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv-bn-swish block.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv(x)
        x = self.bn(x)
        if self.swish is not None:
            x = self.swish(x)
        return x


class SEBlock(Module):
    """Squeeze-and-Excitation block.
    
    Attributes:
        avg_pool: Global average pooling layer.
        se_reduce: Squeeze convolution layer.
        se_expand: Excitation convolution layer.
        swish: Swish activation.
    """

    def __init__(
        self,
        in_channels: int,
        squeezed_channels: int,
    ) -> None:
        """Initialize SE block.
        
        Args:
            in_channels: Number of input/output channels.
            squeezed_channels: Number of channels after squeeze.
        """
        super(SEBlock, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.se_reduce = Conv2d(in_channels, squeezed_channels, kernel_size=1, bias=True)
        self.swish = MemoryEfficientSwish()
        self.se_expand = Conv2d(squeezed_channels, in_channels, kernel_size=1, bias=True)
        self.sigmoid = Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Attention-weighted tensor of same shape.
        """
        scale = self.avg_pool(x)
        scale = self.se_reduce(scale)
        scale = self.swish(scale)
        scale = self.se_expand(scale)
        scale = self.sigmoid(scale)
        return x * scale


class MBConvBlock(Module):
    """Mobile Inverted Residual Bottleneck Block (MBConv).
    
    Attributes:
        has_se: Whether to use SE attention.
        id_skip: Whether to use skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        bn_mom: float = 0.01,
        bn_eps: float = 1e-3,
        id_skip: bool = True,
        tf_mode: bool = False,
    ) -> None:
        """Initialize MBConv block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size for depthwise conv (3 or 5).
            stride: Stride for depthwise conv.
            expand_ratio: Channel expansion factor.
            se_ratio: SE squeeze ratio. Defaults to 0.25.
            bn_mom: BatchNorm momentum. Defaults to 0.01.
            bn_eps: BatchNorm epsilon. Defaults to 1e-3.
            id_skip: Whether to use skip connection. Defaults to True.
            tf_mode: Whether to use TF-style padding. Defaults to False.
        """
        super(MBConvBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.tf_mode = tf_mode
        self.expand_ratio = expand_ratio
        self.id_skip = id_skip
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Whether to use residual connection
        self.use_residual = id_skip and (stride == 1) and (in_channels == out_channels)
        
        # SE configuration
        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        
        mid_channels = in_channels * expand_ratio

        # Expansion phase (1x1 conv) - only if expand_ratio != 1
        if expand_ratio != 1:
            self.expand_conv = Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
            self.bn0 = BatchNorm2d(mid_channels, momentum=bn_mom, eps=bn_eps)
        
        # Depthwise phase
        self.depthwise_conv = Conv2d(
            mid_channels, mid_channels, kernel_size=kernel_size,
            stride=stride, padding=(0 if tf_mode else kernel_size // 2),
            groups=mid_channels, bias=False
        )
        self.bn1 = BatchNorm2d(mid_channels, momentum=bn_mom, eps=bn_eps)
        
        # SE phase
        if self.has_se:
            num_squeezed_channels = max(1, int(in_channels * se_ratio))
            self.se = SEBlock(mid_channels, num_squeezed_channels)
        
        # Projection phase (1x1 conv, no activation)
        self.project_conv = Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = BatchNorm2d(out_channels, momentum=bn_mom, eps=bn_eps)
        
        # Swish activation
        self.swish = MemoryEfficientSwish()

    def forward(self, x: torch.Tensor, drop_connect_rate: Optional[float] = None) -> torch.Tensor:
        """Forward pass through MBConv block.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            drop_connect_rate: Drop connect rate for stochastic depth.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        identity = x
        
        # Expansion
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            x = self.bn0(x)
            x = self.swish(x)
        
        # Depthwise
        if self.tf_mode:
            x = F.pad(x, pad=calc_tf_padding(x, kernel_size=self.kernel_size, stride=self.stride))
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.swish(x)
        
        # SE attention
        if self.has_se:
            x = self.se(x)
        
        # Projection (no activation after this)
        x = self.project_conv(x)
        x = self.bn2(x)
        
        # Skip connection and drop connect
        if self.use_residual:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + identity
        return x
    
    def set_swish(self, memory_efficient: bool = True) -> None:
        """Sets swish function as memory efficient (for training) or standard (for export).
        
        Args:
            memory_efficient: Whether to use memory-efficient version of swish.
        """
        self.swish = MemoryEfficientSwish() if memory_efficient else Swish()
        if self.has_se:
            self.se.swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNetV1(Module):
    """EfficientNet backbone for face recognition.
    
    Attributes:
        features: Sequential container of all feature extraction layers.
        output_layer: Output head for generating face embeddings.
    """

    def __init__(
        self,
        channels: List[List[int]],
        init_block_channels: int,
        final_block_channels: int,
        kernel_sizes: List[List[int]],
        strides_per_stage: List[int],
        expansion_factors: List[List[int]],
        input_size: List[int] = [112, 112],
        num_features: int = 512,
        dropout_rate: float = 0.2,
        drop_connect_rate: float = 0.2,
        tf_mode: bool = False,
        bn_mom: float = 0.01,
        bn_eps: float = 1e-3,
        se_ratio: float = 0.25,
    ) -> None:
        """Initialize EfficientNet backbone.
        
        Args:
            channels: Number of output channels for each unit in each stage.
            init_block_channels: Number of channels in the initial conv block.
            final_block_channels: Number of channels in the final conv block.
            kernel_sizes: Kernel sizes for each unit in each stage.
            strides_per_stage: Stride for the first unit of each stage.
            expansion_factors: Expansion factors for each unit in each stage.
            input_size: Input image size as [H, W]. Defaults to [112, 112].
            num_features: Output embedding dimension. Defaults to 512.
            dropout_rate: Dropout probability. Defaults to 0.2.
            drop_connect_rate: Drop connect rate. Defaults to 0.2.
            tf_mode: Whether to use TF-style padding. Defaults to False.
            bn_mom: BatchNorm momentum. Defaults to 0.01.
            bn_eps: BatchNorm epsilon. Defaults to 1e-3.
            se_ratio: SE squeeze ratio. Defaults to 0.25.
        """
        super(EfficientNetV1, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        
        self.input_size = input_size
        self.drop_connect_rate = drop_connect_rate

        # Stem: 3x3 conv with stride=1 (modified for 112x112 face input)
        self.conv_stem = Conv2d(3, init_block_channels, kernel_size=3, stride=1, 
                                padding=(0 if tf_mode else 1), bias=False)
        self.bn0 = BatchNorm2d(init_block_channels, momentum=bn_mom, eps=bn_eps)
        self.swish = MemoryEfficientSwish()
        
        # Build blocks
        self.blocks = nn.ModuleList([])
        in_channels = init_block_channels
        
        # Count total blocks for drop connect rate scaling
        total_blocks = sum(len(stage_channels) for stage_channels in channels)
        block_idx = 0
        
        for i, channels_per_stage in enumerate(channels):
            kernel_sizes_per_stage = kernel_sizes[i]
            expansion_factors_per_stage = expansion_factors[i]
            
            for j, out_channels in enumerate(channels_per_stage):
                kernel_size = kernel_sizes_per_stage[j]
                expand_ratio = expansion_factors_per_stage[j]
                stride = strides_per_stage[i] if (j == 0) else 1
                
                self.blocks.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        expand_ratio=expand_ratio,
                        se_ratio=se_ratio,
                        bn_mom=bn_mom,
                        bn_eps=bn_eps,
                        id_skip=True,
                        tf_mode=tf_mode,
                    )
                )
                in_channels = out_channels
                block_idx += 1
        
        self.total_blocks = total_blocks
        
        # Head: 1x1 conv to final_block_channels (1280)
        self.conv_head = Conv2d(in_channels, final_block_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(final_block_channels, momentum=bn_mom, eps=bn_eps)
        
        # For 224x224 input: after 4 stride-2 stages -> 14x14
        if input_size[0] == 112:
            out_h, out_w = 7, 7
        else:
            out_h, out_w = 14, 14
        
        self.output_layer = Sequential(
            BatchNorm2d(final_block_channels),
            Dropout(dropout_rate),
            Flatten(),
            Linear(final_block_channels * out_h * out_w, num_features),
            BatchNorm1d(num_features),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights using Kaiming initialization.
        
        """
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (BatchNorm2d, BatchNorm1d)):
                if m.weight is not None:
                    init.constant_(m.weight, 1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def set_swish(self, memory_efficient: bool = True) -> None:
        """Sets swish function as memory efficient (for training) or standard (for export).
        
        Args:
            memory_efficient: Whether to use memory-efficient version of swish.
        """
        self.swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self.blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using convolution layers.
        
        Args:
            x: Input tensor of shape (N, 3, H, W).
            
        Returns:
            torch.Tensor: Feature tensor.
        """
        # Stem
        x = self.swish(self.bn0(self.conv_stem(x)))
        
        # Blocks with drop connect
        for idx, block in enumerate(self.blocks):
            drop_rate = self.drop_connect_rate
            if drop_rate:
                drop_rate *= float(idx) / self.total_blocks  # Scale drop connect rate
            x = block(x, drop_connect_rate=drop_rate)
        
        # Head
        x = self.swish(self.bn1(self.conv_head(x)))
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embeddings from input images.
        
        Args:
            x: Input images of shape (N, 3, H, W).
            
        Returns:
            torch.Tensor: Face embeddings of shape (N, num_features).
        """
        x = self.extract_features(x)
        x = self.output_layer(x)
        return x


def _create_efficientnet(
    version: str,
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.2,
    drop_connect_rate: float = 0.2,
    **kwargs,
) -> EfficientNetV1:
    """Create EfficientNet model with specific version parameters.

    Args:
        version: Model version ('b0' to 'b7').
        input_size: Input image size. Defaults to [112, 112].
        num_features: Output embedding dimension. Defaults to 512.
        dropout_rate: Dropout probability. Defaults to 0.2.
        drop_connect_rate: Drop connect rate. Defaults to 0.2.
        
    Returns:
        EfficientNetV1: Configured EfficientNet model.
        
    Raises:
        ValueError: If unsupported version is specified.
    """
    if version.endswith("b") or version.endswith("c"):
        version = version[:-1]
        tf_mode = True
    else:
        tf_mode = False
    
    bn_mom = 0.01  # PyTorch convention: 1 - 0.99 (TF)
    bn_eps = 1e-3
    
    version_configs = {
        "b0": {"depth_factor": 1.0, "width_factor": 1.0, "dropout": 0.2},
        "b1": {"depth_factor": 1.1, "width_factor": 1.0, "dropout": 0.2},
        "b2": {"depth_factor": 1.2, "width_factor": 1.1, "dropout": 0.3},
        "b3": {"depth_factor": 1.4, "width_factor": 1.2, "dropout": 0.3},
        "b4": {"depth_factor": 1.8, "width_factor": 1.4, "dropout": 0.4},
        "b5": {"depth_factor": 2.2, "width_factor": 1.6, "dropout": 0.4},
        "b6": {"depth_factor": 2.6, "width_factor": 1.8, "dropout": 0.5},
        "b7": {"depth_factor": 3.1, "width_factor": 2.0, "dropout": 0.5},
    }
    
    if version not in version_configs:
        raise ValueError(f"Unsupported EfficientNet version: {version}")
    
    config = version_configs[version]
    depth_factor = config["depth_factor"]
    width_factor = config["width_factor"]
    default_dropout = config["dropout"]
    
    
    init_block_channels = 32
    layers = [1, 2, 2, 3, 3, 4, 1]
    downsample = [1, 1, 1, 1, 0, 1, 0]  # which stages start new downsampling groups
    channels_per_layers = [16, 24, 40, 80, 112, 192, 320]
    expansion_factors_per_layers = [1, 6, 6, 6, 6, 6, 6]
    kernel_sizes_per_layers = [3, 3, 5, 3, 5, 5, 3]
    strides_per_stage = [1, 2, 2, 2, 1, 2, 1]
    final_block_channels = 1280

    # Apply depth scaling
    layers = [int(math.ceil(li * depth_factor)) for li in layers]
    
    # Apply width scaling
    channels_per_layers = [round_channels(ci * width_factor) for ci in channels_per_layers]
    init_block_channels = round_channels(init_block_channels * width_factor)
    final_block_channels = round_channels(final_block_channels * width_factor)

    from functools import reduce
    
    def merge_stages(x, y):
        if y[2] != 0:
            return x + [[y[0]] * y[1]]
        else:
            return x[:-1] + [x[-1] + [y[0]] * y[1]]
    
    channels = reduce(
        merge_stages,
        zip(channels_per_layers, layers, downsample),
        [],
    )
    kernel_sizes = reduce(
        merge_stages,
        zip(kernel_sizes_per_layers, layers, downsample),
        [],
    )
    expansion_factors = reduce(
        merge_stages,
        zip(expansion_factors_per_layers, layers, downsample),
        [],
    )
    strides = reduce(
        merge_stages,
        zip(strides_per_stage, layers, downsample),
        [],
    )
    strides_per_stage = [si[0] for si in strides]

    return EfficientNetV1(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernel_sizes=kernel_sizes,
        strides_per_stage=strides_per_stage,
        expansion_factors=expansion_factors,
        input_size=input_size,
        num_features=num_features,
        dropout_rate=dropout_rate if dropout_rate != 0.2 else default_dropout,
        drop_connect_rate=drop_connect_rate,
        tf_mode=tf_mode,
        bn_mom=bn_mom,
        bn_eps=bn_eps,
        se_ratio=0.25,
        **kwargs,
    )


def EfficientNetV1_B0(**kwargs) -> EfficientNetV1:
    """Construct an EfficientNet-B0 model.
    
    Baseline model with width_factor=1.0, depth_factor=1.0.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.2.
        
    Returns:
        EfficientNetV1: EfficientNet-B0 model instance.
    """
    return _create_efficientnet("b0", **kwargs)


def EfficientNetV1_B1(**kwargs) -> EfficientNetV1:
    """Construct an EfficientNet-B1 model.
    
    Scaled model with width_factor=1.0, depth_factor=1.1.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.2.
        
    Returns:
        EfficientNetV1: EfficientNet-B1 model instance.
    """
    return _create_efficientnet("b1", **kwargs)


def EfficientNetV1_B2(**kwargs) -> EfficientNetV1:
    """Construct an EfficientNet-B2 model.
    
    Scaled model with width_factor=1.1, depth_factor=1.2.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.3.
        
    Returns:
        EfficientNetV1: EfficientNet-B2 model instance.
    """
    return _create_efficientnet("b2", **kwargs)


def EfficientNetV1_B3(**kwargs) -> EfficientNetV1:
    """Construct an EfficientNet-B3 model.
    
    Scaled model with width_factor=1.2, depth_factor=1.4.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.3.
        
    Returns:
        EfficientNetV1: EfficientNet-B3 model instance.
    """
    return _create_efficientnet("b3", **kwargs)


def EfficientNetV1_B4(**kwargs) -> EfficientNetV1:
    """Construct an EfficientNet-B4 model.
    
    Scaled model with width_factor=1.4, depth_factor=1.8.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        
    Returns:
        EfficientNetV1: EfficientNet-B4 model instance.
    """
    return _create_efficientnet("b4", **kwargs)


def EfficientNetV1_B5(**kwargs) -> EfficientNetV1:
    """Construct an EfficientNet-B5 model.
    
    Scaled model with width_factor=1.6, depth_factor=2.2.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        
    Returns:
        EfficientNetV1: EfficientNet-B5 model instance.
    """
    return _create_efficientnet("b5", **kwargs)


def EfficientNetV1_B6(**kwargs) -> EfficientNetV1:
    """Construct an EfficientNet-B6 model.
    
    Scaled model with width_factor=1.8, depth_factor=2.6.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.5.
        
    Returns:
        EfficientNetV1: EfficientNet-B6 model instance.
    """
    return _create_efficientnet("b6", **kwargs)


def EfficientNetV1_B7(**kwargs) -> EfficientNetV1:
    """Construct an EfficientNet-B7 model.
    
    Largest scaled model with width_factor=2.0, depth_factor=3.1.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.5.
        
    Returns:
        EfficientNetV1: EfficientNet-B7 model instance.
    """
    return _create_efficientnet("b7", **kwargs)
