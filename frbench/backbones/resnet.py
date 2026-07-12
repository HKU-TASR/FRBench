"""
Standard ResNet backbone
Reference Paper: Deep Residual Learning for Image Recognition
Reference Code: https://github.com/cavalleria/cavaface/blob/master/backbone/resnet.py

Supported Variants: ResNet-18, ResNet-34, ResNet-50, ResNet-100, ResNet-101, ResNet-152, ResNet-200
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

from .utils import conv1x1, conv3x3

__all__ = [
    "ResNet",
    "ResNet_18",
    "ResNet_34",
    "ResNet_50",
    "ResNet_100",
    "ResNet_101",
    "ResNet_152",
    "ResNet_200",
]

class BasicBlock(Module):
    """Basic residual block for ResNet-18 and ResNet-34.
    
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
    ) -> None:
        """Initialize BasicBlock.
        
        Args:
            inplanes: Number of input channels.
            planes: Number of output channels (before expansion).
            stride: Stride for the first convolution. Defaults to 1.
            downsample: Downsample module for identity mapping. Defaults to None.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the basic block.
        
        Args:
            x: Input tensor of shape (N, inplanes, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, planes, H', W').
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(Module):
    """Bottleneck residual block for ResNet-50 and deeper variants.
    
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
    ) -> None:
        """Initialize Bottleneck.
        
        Args:
            inplanes: Number of input channels.
            planes: Number of intermediate channels.
            stride: Stride for the 3x3 convolution. Defaults to 1.
            downsample: Downsample module for identity mapping. Defaults to None.
        """
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.relu = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the bottleneck block.
        
        Args:
            x: Input tensor of shape (N, inplanes, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, planes*4, H', W').
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(Module):
    """Standard ResNet backbone for face recognition.
    
    Attributes:
        num_features: Output embedding dimension.
        input_size: Input image size as [H, W].
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        input_size: List[int] = [112, 112],
        num_features: int = 512,
        dropout_rate: float = 0.4,
        zero_init_residual: bool = True,
    ) -> None:
        """Initialize ResNet backbone.
        
        Keyword Args:
            input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
                or [224, 224]. Defaults to [112, 112].
            num_features (int): Output embedding dimension. Defaults to 512.
            dropout_rate (float): Dropout probability before the final FC layer. 
                Defaults to 0.4.
            zero_init_residual (bool): Zero-initialize last BN in each residual 
                branch for better convergence. Defaults to True.
        
        Internal Args (set by factory functions, not user-configurable):
            block: Residual block class (BasicBlock or Bottleneck).
            layers: Number of blocks in each of the 4 stages.
        """
        super(ResNet, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        
        self.inplanes = 64
        self.num_features = num_features
        
        # Initial convolution layer
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Output layers
        final_planes = 512 * block.expansion
        self.bn_o1 = BatchNorm2d(final_planes)
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
            zero_init_residual: If True, zero-initialize the last BN in each
                residual branch for better convergence.
        """
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (BatchNorm2d, BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> Sequential:
        """Create a residual stage with multiple blocks.
        
        Args:
            block: Residual block class.
            planes: Number of output channels for each block.
            blocks: Number of blocks in this stage.
            stride: Stride for the first block. Defaults to 1.
            
        Returns:
            Sequential: A sequential container of residual blocks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embeddings from input images.
        
        Args:
            x: Input images of shape (N, 3, H, W).
            
        Returns:
            torch.Tensor: Face embeddings of shape (N, num_features).
        """
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Output layers
        x = self.bn_o1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn_o2(x)

        return x


def ResNet_18(**kwargs) -> ResNet:
    """Construct a ResNet-18 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        ResNet: ResNet-18 model instance.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet_34(**kwargs) -> ResNet:
    """Construct a ResNet-34 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        ResNet: ResNet-34 model instance.
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet_50(**kwargs) -> ResNet:
    """Construct a ResNet-50 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        ResNet: ResNet-50 model instance.
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet_100(**kwargs) -> ResNet:
    """Construct a ResNet-100 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        ResNet: ResNet-100 model instance.
    """
    return ResNet(Bottleneck, [3, 13, 30, 3], **kwargs)


def ResNet_101(**kwargs) -> ResNet:
    """Construct a ResNet-101 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        ResNet: ResNet-101 model instance.
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet_152(**kwargs) -> ResNet:
    """Construct a ResNet-152 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        ResNet: ResNet-152 model instance.
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def ResNet_200(**kwargs) -> ResNet:
    """Construct a ResNet-200 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        ResNet: ResNet-200 model instance.
    """
    return ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)

