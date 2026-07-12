"""
Standard ResNetV2 backbone
Reference Paper: Identity Mappings in Deep Residual Networks
Reference Code: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnetv2.py

Supported Variants: ResNetV2-18, ResNetV2-34, ResNetV2-50, ResNetV2-100, ResNetV2-101, ResNetV2-152, ResNetV2-200
"""
from typing import Type, Union, List, Optional

import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm1d,
    BatchNorm2d,
    ReLU,
    Dropout,
    MaxPool2d,
    Linear,
)

from .utils import conv1x1, conv3x3, DropPath


# Downsample modules
class DownsampleConv(Module):
    """1x1 convolution downsampling module (standard ResNet style)."""

    def __init__(self, in_chs: int, out_chs: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = conv1x1(in_chs, out_chs, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownsampleAvg(Module):
    """AvgPool downsampling module (ResNet-D style).
    
    Uses average pooling before 1x1 conv to reduce information loss.
    """

    def __init__(self, in_chs: int, out_chs: int, stride: int = 1) -> None:
        super().__init__()
        if stride > 1:
            self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()
        self.conv = conv1x1(in_chs, out_chs, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


__all__ = [
    "ResNetV2",
    "ResNetV2_18",
    "ResNetV2_34",
    "ResNetV2_50",
    "ResNetV2_100",
    "ResNetV2_101",
    "ResNetV2_152",
    "ResNetV2_200",
]


class PreActBasicBlock(Module):
    """Pre-activation basic block for ResNetV2.
    
    Applies BN and ReLU before convolution for improved gradient flow.
    
    Attributes:
        expansion: Output channel expansion factor (1 for BasicBlock).
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[Module] = None,
        drop_path_rate: float = 0.0,
    ) -> None:
        """Initialize PreActBasicBlock.
        
        Args:
            inplanes: Number of input channels.
            planes: Number of output channels.
            stride: Stride for the first convolution. Defaults to 1.
            downsample: Downsample module for identity mapping. Defaults to None.
            drop_path_rate: Stochastic depth drop rate. Defaults to 0.0.
        """
        super(PreActBasicBlock, self).__init__()
        self.bn1 = BatchNorm2d(inplanes)
        self.relu = ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the pre-activation basic block.
        
        Args:
            x: Input tensor of shape (N, inplanes, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, planes, H', W').
        """
        out = self.bn1(x)
        out = self.relu(out)
        
        # Apply downsample to pre-activated input
        if self.downsample is not None:
            identity = self.downsample(out)
        else:
            identity = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.drop_path(out)

        out += identity
        return out


class PreActBottleneck(Module):
    """Pre-activation bottleneck block for ResNetV2.
    
    Applies BN and ReLU before convolution for improved gradient flow.
    
    Attributes:
        expansion: Output channel expansion factor (4 for Bottleneck).
    """
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[Module] = None,
        drop_path_rate: float = 0.0,
    ) -> None:
        """Initialize PreActBottleneck.
        
        Args:
            inplanes: Number of input channels.
            planes: Number of intermediate channels.
            stride: Stride for the 3x3 convolution. Defaults to 1.
            downsample: Downsample module for identity mapping. Defaults to None.
            drop_path_rate: Stochastic depth drop rate. Defaults to 0.0.
        """
        super(PreActBottleneck, self).__init__()
        self.bn1 = BatchNorm2d(inplanes)
        self.relu = ReLU(inplace=True)
        self.conv1 = conv1x1(inplanes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the pre-activation bottleneck block.
        
        Args:
            x: Input tensor of shape (N, inplanes, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, planes*4, H', W').
        """
        out = self.bn1(x)
        out = self.relu(out)
        
        # Apply downsample to pre-activated input
        if self.downsample is not None:
            identity = self.downsample(out)
        else:
            identity = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.drop_path(out)

        out += identity
        return out


class ResNetV2(Module):
    """Pre-activation ResNet (ResNetV2) backbone for face recognition.
    
    Uses pre-activation design where BN and ReLU precede convolution.
    
    Attributes:
        num_features: Output embedding dimension.
        input_size: Input image size as [H, W].
    """

    def __init__(
        self,
        block: Type[Union[PreActBasicBlock, PreActBottleneck]],
        layers: List[int],
        input_size: List[int] = [112, 112],
        num_features: int = 512,
        dropout_rate: float = 0.4,
        drop_path_rate: float = 0.0,
        zero_init_residual: bool = False,
        avg_down: bool = False,
    ) -> None:
        """Initialize ResNetV2 backbone.
        
        Keyword Args:
            input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
                or [224, 224]. Defaults to [112, 112].
            num_features (int): Output embedding dimension. Defaults to 512.
            dropout_rate (float): Dropout probability before the final FC layer. 
                Defaults to 0.4.
            drop_path_rate (float): Stochastic depth drop rate, linearly increased
                from 0 to this value across all blocks. Defaults to 0.0.
            zero_init_residual (bool): Zero-initialize last conv in each residual 
                branch for better convergence. Defaults to True.
            avg_down (bool): Use average pooling in residual downsampling 
                (ResNet-D style) instead of strided convolution. Defaults to False.
        
        Internal Args (set by factory functions, not user-configurable):
            block: Residual block class (PreActBasicBlock or PreActBottleneck).
            layers: Number of blocks in each of the 4 stages.
        """
        super(ResNetV2, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        
        self.inplanes = 64
        self.num_features = num_features
        self.drop_path_rate = drop_path_rate
        self.avg_down = avg_down
        
        # Initial convolution layer (pure V2 style: no BN+ReLU before first block)
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Calculate total number of blocks for drop path rate scheduling
        total_blocks = sum(layers)
        block_idx = 0
        
        # Residual layers with linearly increasing drop path rates
        self.layer1, block_idx = self._make_layer(block, 64, layers[0], block_idx=block_idx, total_blocks=total_blocks)
        self.layer2, block_idx = self._make_layer(block, 128, layers[1], stride=2, block_idx=block_idx, total_blocks=total_blocks)
        self.layer3, block_idx = self._make_layer(block, 256, layers[2], stride=2, block_idx=block_idx, total_blocks=total_blocks)
        self.layer4, block_idx = self._make_layer(block, 512, layers[3], stride=2, block_idx=block_idx, total_blocks=total_blocks)
        
        # Final batch norm (important for pre-activation networks)
        # This completes the pre-activation for the last residual block's output
        final_planes = 512 * block.expansion
        self.bn_final = BatchNorm2d(final_planes)
        self.relu = ReLU(inplace=True)
        
        # Output layers
        self.dropout = Dropout(p=dropout_rate)
        
        # Calculate feature map size based on input size
        if input_size[0] == 112:
            feature_size = 4 * 4
        else:  # 224
            feature_size = 8 * 8
        
        self.fc = Linear(final_planes * feature_size, num_features)
        self.bn_o2 = BatchNorm1d(num_features)

        # Weight initialization
        self._initialize_weights(zero_init_residual)

    def _initialize_weights(self, zero_init_residual: bool) -> None:
        """Initialize model weights using Kaiming initialization.
        
        Args:
            zero_init_residual: If True, zero-initialize the last conv weight in each
                residual branch so the block starts as identity.
        """
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (BatchNorm2d, BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last conv weight in each residual branch.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PreActBottleneck):
                    nn.init.constant_(m.conv3.weight, 0)
                elif isinstance(m, PreActBasicBlock):
                    nn.init.constant_(m.conv2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[PreActBasicBlock, PreActBottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        block_idx: int = 0,
        total_blocks: int = 1,
    ) -> tuple:
        """Create a residual stage with multiple pre-activation blocks.
        
        Args:
            block: Residual block class.
            planes: Number of output channels for each block.
            blocks: Number of blocks in this stage.
            stride: Stride for the first block. Defaults to 1.
            block_idx: Current block index for drop path rate scheduling.
            total_blocks: Total number of blocks in the network.
            
        Returns:
            tuple: (Sequential container of residual blocks, updated block_idx).
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # For pre-activation, downsample is applied after BN+ReLU
            # Use DownsampleAvg (ResNet-D style) or DownsampleConv based on avg_down setting
            proj_layer = DownsampleAvg if self.avg_down else DownsampleConv
            downsample = proj_layer(self.inplanes, planes * block.expansion, stride)

        layers = []
        # Linearly increase drop path rate
        dpr = self.drop_path_rate * block_idx / (total_blocks - 1) if total_blocks > 1 else 0.0
        layers.append(block(self.inplanes, planes, stride, downsample, drop_path_rate=dpr))
        block_idx += 1
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            dpr = self.drop_path_rate * block_idx / (total_blocks - 1) if total_blocks > 1 else 0.0
            layers.append(block(self.inplanes, planes, drop_path_rate=dpr))
            block_idx += 1

        return Sequential(*layers), block_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embeddings from input images.
        
        Args:
            x: Input images of shape (N, 3, H, W).
            
        Returns:
            torch.Tensor: Face embeddings of shape (N, num_features).
        """
        # Initial layers (pure V2 style: no BN+ReLU, pre-activation happens in first block)
        x = self.conv1(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Final pre-activation (completes the last block's pre-act pattern)
        x = self.bn_final(x)
        x = self.relu(x)

        # Output layers
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn_o2(x)

        return x


def ResNetV2_18(**kwargs) -> ResNetV2:
    """Construct a ResNetV2-18 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        ResNetV2: ResNetV2-18 model instance.
    """
    return ResNetV2(PreActBasicBlock, [2, 2, 2, 2], **kwargs)


def ResNetV2_34(**kwargs) -> ResNetV2:
    """Construct a ResNetV2-34 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        ResNetV2: ResNetV2-34 model instance.
    """
    return ResNetV2(PreActBasicBlock, [3, 4, 6, 3], **kwargs)


def ResNetV2_50(**kwargs) -> ResNetV2:
    """Construct a ResNetV2-50 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        ResNetV2: ResNetV2-50 model instance.
    """
    return ResNetV2(PreActBottleneck, [3, 4, 6, 3], **kwargs)


def ResNetV2_100(**kwargs) -> ResNetV2:
    """Construct a ResNetV2-100 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        ResNetV2: ResNetV2-100 model instance.
    """
    return ResNetV2(PreActBottleneck, [3, 13, 30, 3], **kwargs)


def ResNetV2_101(**kwargs) -> ResNetV2:
    """Construct a ResNetV2-101 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        ResNetV2: ResNetV2-101 model instance.
    """
    return ResNetV2(PreActBottleneck, [3, 4, 23, 3], **kwargs)


def ResNetV2_152(**kwargs) -> ResNetV2:
    """Construct a ResNetV2-152 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        ResNetV2: ResNetV2-152 model instance.
    """
    return ResNetV2(PreActBottleneck, [3, 8, 36, 3], **kwargs)


def ResNetV2_200(**kwargs) -> ResNetV2:
    """Construct a ResNetV2-200 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        ResNetV2: ResNetV2-200 model instance.
    """
    return ResNetV2(PreActBottleneck, [3, 24, 36, 3], **kwargs)