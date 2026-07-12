"""
Improved ResNet with Squeeze-and-Excitation (IR-SE) backbone from ArcFace and SE-Net
Reference Paper: Squeeze-and-Excitation Networks
Reference Code: https://github.com/cavalleria/cavaface.pytorch/blob/master/backbone/resnet_irse.py

Supported Variants: IR_SE-50, IR_SE-100, IR_SE-101, IR_SE-152, IR_SE-200
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
    ReLU,
    Dropout,
    Linear,
    Sigmoid,
    MaxPool2d,
    AdaptiveAvgPool2d,
)


__all__ = [
    "IR_SE",
    "IR_SE_18",
    "IR_SE_34",
    "IR_SE_50",
    "IR_SE_100",
    "IR_SE_101",
    "IR_SE_152",
    "IR_SE_185",
    "IR_SE_200",
]


class SEModule(Module):
    """Squeeze-and-Excitation (SE) attention module.
    
    Adaptively recalibrates channel-wise feature responses.
    Follows cavaface implementation exactly.
    
    Attributes:
        avg_pool: Global average pooling layer.
        fc1: First fully-connected layer (squeeze).
        fc2: Second fully-connected layer (excitation).
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize SE module.
        
        Args:
            channels: Number of input/output channels.
            reduction: Channel reduction ratio for bottleneck. Defaults to 16.
        """
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        # Xavier initialization for fc1 (matches cavaface)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input features.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Recalibrated tensor of shape (N, C, H, W).
        """
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BottleneckIRSE(Module):
    """IR-SE bottleneck block following cavaface implementation.
    
    Structure: BN -> Conv3x3(stride=1) -> PReLU -> Conv3x3(stride) -> BN -> SE
    
    Note: Despite the name "bottleneck", this uses two 3x3 convs like BasicBlock,
    not the 1x1-3x3-1x1 structure. This matches the cavaface implementation.
    """

    def __init__(self, in_channel: int, depth: int, stride: int = 1) -> None:
        """Initialize BottleneckIRSE.
        
        Args:
            in_channel: Number of input channels.
            depth: Number of output channels.
            stride: Stride for the second convolution. Defaults to 1.
        """
        super(BottleneckIRSE, self).__init__()
        # Shortcut layer
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        # Residual layer
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the IR-SE block.
        
        Args:
            x: Input tensor of shape (N, in_channel, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, depth, H', W').
        """
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


def _get_blocks(num_layers: int) -> List[List[tuple]]:
    """Get block configurations for different network depths.
    
    Args:
        num_layers: Number of layers (18, 34, 50, 100, 101, 152, 200).
        
    Returns:
        List of block configurations, each as (in_channel, depth, stride).
    """
    if num_layers == 18:
        blocks = [
            [(64, 64, 2)] + [(64, 64, 1)] * 1,      # 2 blocks
            [(64, 128, 2)] + [(128, 128, 1)] * 1,   # 2 blocks
            [(128, 256, 2)] + [(256, 256, 1)] * 1,  # 2 blocks
            [(256, 512, 2)] + [(512, 512, 1)] * 1,  # 2 blocks
        ]
    elif num_layers == 34:
        blocks = [
            [(64, 64, 2)] + [(64, 64, 1)] * 2,      # 3 blocks
            [(64, 128, 2)] + [(128, 128, 1)] * 3,   # 4 blocks
            [(128, 256, 2)] + [(256, 256, 1)] * 5,  # 6 blocks
            [(256, 512, 2)] + [(512, 512, 1)] * 2,  # 3 blocks
        ]
    elif num_layers == 50:
        blocks = [
            [(64, 64, 2)] + [(64, 64, 1)] * 2,      # 3 blocks
            [(64, 128, 2)] + [(128, 128, 1)] * 3,   # 4 blocks
            [(128, 256, 2)] + [(256, 256, 1)] * 13, # 14 blocks
            [(256, 512, 2)] + [(512, 512, 1)] * 2,  # 3 blocks
        ]
    elif num_layers == 100:
        blocks = [
            [(64, 64, 2)] + [(64, 64, 1)] * 2,      # 3 blocks
            [(64, 128, 2)] + [(128, 128, 1)] * 12,  # 13 blocks
            [(128, 256, 2)] + [(256, 256, 1)] * 29, # 30 blocks
            [(256, 512, 2)] + [(512, 512, 1)] * 2,  # 3 blocks
        ]
    elif num_layers == 101:
        blocks = [
            [(64, 64, 2)] + [(64, 64, 1)] * 2,      # 3 blocks
            [(64, 128, 2)] + [(128, 128, 1)] * 3,   # 4 blocks
            [(128, 256, 2)] + [(256, 256, 1)] * 22, # 23 blocks
            [(256, 512, 2)] + [(512, 512, 1)] * 2,  # 3 blocks
        ]
    elif num_layers == 152:
        blocks = [
            [(64, 64, 2)] + [(64, 64, 1)] * 2,      # 3 blocks
            [(64, 128, 2)] + [(128, 128, 1)] * 7,   # 8 blocks
            [(128, 256, 2)] + [(256, 256, 1)] * 35, # 36 blocks
            [(256, 512, 2)] + [(512, 512, 1)] * 2,  # 3 blocks
        ]
    elif num_layers == 185:
        blocks = [
            [(64, 64, 2)] + [(64, 64, 1)] * 2,      # 3 blocks
            [(64, 128, 2)] + [(128, 128, 1)] * 21,  # 22 blocks
            [(128, 256, 2)] + [(256, 256, 1)] * 32, # 33 blocks
            [(256, 512, 2)] + [(512, 512, 1)] * 2,  # 3 blocks
        ]
    elif num_layers == 200:
        blocks = [
            [(64, 64, 2)] + [(64, 64, 1)] * 2,      # 3 blocks
            [(64, 128, 2)] + [(128, 128, 1)] * 23,  # 24 blocks
            [(128, 256, 2)] + [(256, 256, 1)] * 35, # 36 blocks
            [(256, 512, 2)] + [(512, 512, 1)] * 2,  # 3 blocks
        ]
    else:
        raise ValueError(f"Unsupported num_layers: {num_layers}. Choose from 18, 34, 50, 100, 101, 152, 185, 200.")
    
    return blocks


class IR_SE(Module):
    """IR backbone with Squeeze-and-Excitation attention.
    
    Follows cavaface implementation exactly.
    
    Attributes:
        num_features: Output embedding dimension.
        input_size: Input image size as [H, W].
    """

    def __init__(
        self,
        num_layers: int,
        input_size: List[int] = [112, 112],
        num_features: int = 512,
        dropout_rate: float = 0.4,
    ) -> None:
        """Initialize IR-SE backbone.
        
        Args:
            num_layers: Number of layers (50, 100, 101, 152, 200).
            input_size: Input image size as [H, W]. Supports [112, 112] 
                or [224, 224]. Defaults to [112, 112].
            num_features: Output embedding dimension. Defaults to 512.
            dropout_rate: Dropout probability before the final FC layer. 
                Defaults to 0.4.
        """
        super(IR_SE, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 101, 152, 185, 200], "num_layers should be 18, 34, 50, 100, 101, 152, 185, 200"
        
        self.num_features = num_features
        blocks = _get_blocks(num_layers)
        
        # Input layer: 3x3 conv (different from standard ResNet's 7x7)
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )
        
        # Build body from blocks
        modules = []
        for stage_blocks in blocks:
            for in_channel, depth, stride in stage_blocks:
                modules.append(BottleneckIRSE(in_channel, depth, stride))
        self.body = Sequential(*modules)
        
        # Output layers
        self.bn2 = BatchNorm2d(512, eps=1e-05)
        self.dropout = Dropout(p=dropout_rate, inplace=True)
        
        # Calculate feature map size based on input size
        # After 4 stride-2 layers: 112 -> 56 -> 28 -> 14 -> 7, 224 -> 112 -> 56 -> 28 -> 14
        if input_size[0] == 112:
            feature_size = 7 * 7
        else:  # 224
            feature_size = 14 * 14
        
        self.fc = Linear(512 * feature_size, num_features)
        self.features = BatchNorm1d(num_features, eps=1e-05)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embeddings from input images.
        
        Args:
            x: Input images of shape (N, 3, H, W).
            
        Returns:
            torch.Tensor: Face embeddings of shape (N, num_features).
        """
        x = self.input_layer(x)
        x = self.body(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


def IR_SE_18(**kwargs) -> IR_SE:
    """Construct an IR_SE-18 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        
    Returns:
        IR_SE: IR_SE-18 model instance.
    """
    return IR_SE(18, **kwargs)


def IR_SE_34(**kwargs) -> IR_SE:
    """Construct an IR_SE-34 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        
    Returns:
        IR_SE: IR_SE-34 model instance.
    """
    return IR_SE(34, **kwargs)


def IR_SE_50(**kwargs) -> IR_SE:
    """Construct an IR_SE-50 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        
    Returns:
        IR_SE: IR_SE-50 model instance.
    """
    return IR_SE(50, **kwargs)


def IR_SE_100(**kwargs) -> IR_SE:
    """Construct an IR_SE-100 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        
    Returns:
        IR_SE: IR_SE-100 model instance.
    """
    return IR_SE(100, **kwargs)


def IR_SE_101(**kwargs) -> IR_SE:
    """Construct an IR_SE-101 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        
    Returns:
        IR_SE: IR_SE-101 model instance.
    """
    return IR_SE(101, **kwargs)


def IR_SE_152(**kwargs) -> IR_SE:
    """Construct an IR_SE-152 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        
    Returns:
        IR_SE: IR_SE-152 model instance.
    """
    return IR_SE(152, **kwargs)


def IR_SE_185(**kwargs) -> IR_SE:
    """Construct an IR_SE-185 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        
    Returns:
        IR_SE: IR_SE-185 model instance.
    """
    return IR_SE(185, **kwargs)


def IR_SE_200(**kwargs) -> IR_SE:
    """Construct an IR_SE-200 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        
    Returns:
        IR_SE: IR_SE-200 model instance.
    """
    return IR_SE(200, **kwargs)