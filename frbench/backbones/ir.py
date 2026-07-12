"""
Improved ResNet (IR) backbone from ArcFace
Reference Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
Reference Code: https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/iresnet.py

Supported Variants: IR-18, IR-34, IR-50, IR-100, IR-101, IR-152, IR-200
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
    PReLU,
    Dropout,
    Linear,
)

from .utils import conv1x1, conv3x3


__all__ = [
    "IR",
    "IR_18",
    "IR_34",
    "IR_50",
    "IR_100",
    "IR_101",
    "IR_152",
    "IR_200",
]

class IRBasicBlock(Module):
    """IR basic block with pre-norm and PReLU activation.
    
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
        """Initialize IRBasicBlock.
        
        Args:
            inplanes: Number of input channels.
            planes: Number of output channels.
            stride: Stride for the first convolution. Defaults to 1.
            downsample: Downsample module for identity mapping. Defaults to None.
        """
        super(IRBasicBlock, self).__init__()
        self.bn1 = BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.prelu = PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the IR basic block.
        
        Args:
            x: Input tensor of shape (N, inplanes, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, planes, H', W').
        """
        identity = x
        
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class IRBottleneck(Module):
    """IR bottleneck block with pre-norm and PReLU activation.
    
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
        """Initialize IRBottleneck.
        
        Args:
            inplanes: Number of input channels.
            planes: Number of intermediate channels.
            stride: Stride for the 3x3 convolution. Defaults to 1.
            downsample: Downsample module for identity mapping. Defaults to None.
        """
        super(IRBottleneck, self).__init__()
        self.bn1 = BatchNorm2d(inplanes)
        self.conv1 = conv1x1(inplanes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.prelu1 = PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = BatchNorm2d(planes)
        self.prelu2 = PReLU(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn4 = BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the IR bottleneck block.
        
        Args:
            x: Input tensor of shape (N, inplanes, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, planes*4, H', W').
        """
        identity = x
        
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.prelu2(out)
        out = self.conv3(out)
        out = self.bn4(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class IR(Module):
    """Improved ResNet (IR) backbone for face recognition.
    
    Uses 3x3 input conv, PReLU activation, and pre-norm style.
    
    Attributes:
        num_features: Output embedding dimension.
        input_size: Input image size as [H, W].
    """

    def __init__(
        self,
        block: Type[Union[IRBasicBlock, IRBottleneck]],
        layers: List[int],
        input_size: List[int] = [112, 112],
        num_features: int = 512,
        dropout_rate: float = 0.0,
        zero_init_residual: bool = True,
    ) -> None:
        """Initialize IR backbone.
        
        Keyword Args:
            input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
                or [224, 224]. Defaults to [112, 112].
            num_features (int): Output embedding dimension. Defaults to 512.
            dropout_rate (float): Dropout probability before the final FC layer. 
                Defaults to 0.4.
            zero_init_residual (bool): Zero-initialize last BN in each residual 
                branch for better convergence. Defaults to True.
        
        Internal Args (set by factory functions, not user-configurable):
            block: Residual block class (IRBasicBlock or IRBottleneck).
            layers: Number of blocks in each of the 4 stages.
        """
        super(IR, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        
        self.inplanes = 64
        self.num_features = num_features
        
        # Input layer: 3x3 conv (different from standard ResNet's 7x7)
        self.input_layer = Sequential(
            Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Output layers
        final_planes = 512 * block.expansion
        self.bn_o1 = BatchNorm2d(final_planes)
        self.dropout = Dropout(p=dropout_rate, inplace=True)
        
        # Calculate feature map size based on input size
        # After 4 stride-2 layers: 112 -> 56 -> 28 -> 14 -> 7, 224 -> 112 -> 56 -> 28 -> 14
        if input_size[0] == 112:
            feature_size = 7 * 7
        else:  # 224
            feature_size = 14 * 14
        
        self.fc = Linear(final_planes * feature_size, num_features)
        self.bn_o2 = BatchNorm1d(num_features, eps=1e-05)

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
                nn.init.normal_(m.weight, 0, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (BatchNorm2d, BatchNorm1d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IRBottleneck):
                    nn.init.constant_(m.bn4.weight, 0)
                elif isinstance(m, IRBasicBlock):
                    nn.init.constant_(m.bn3.weight, 0)  # bn3 is the last BN in residual branch

    def _make_layer(
        self,
        block: Type[Union[IRBasicBlock, IRBottleneck]],
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
        # Input layer
        x = self.input_layer(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Output layers
        x = self.bn_o1(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_o2(x)

        return x


def IR_18(**kwargs) -> IR:
    """Construct an IR-18 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        IR: IR-18 model instance.
    """
    return IR(IRBasicBlock, [2, 2, 2, 2], **kwargs)


def IR_34(**kwargs) -> IR:
    """Construct an IR-34 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        IR: IR-34 model instance.
    """
    return IR(IRBasicBlock, [3, 4, 6, 3], **kwargs)


def IR_50(**kwargs) -> IR:
    """Construct an IR-50 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        IR: IR-50 model instance.
    """
    return IR(IRBasicBlock, [3, 4, 14, 3], **kwargs)


def IR_100(**kwargs) -> IR:
    """Construct an IR-100 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        IR: IR-100 model instance.
    """
    return IR(IRBasicBlock, [3, 13, 30, 3], **kwargs)


def IR_101(**kwargs) -> IR:
    """Construct an IR-101 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        IR: IR-101 model instance.
    """
    return IR(IRBasicBlock, [3, 4, 23, 3], **kwargs)


def IR_152(**kwargs) -> IR:
    """Construct an IR-152 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        IR: IR-152 model instance.
    """
    return IR(IRBasicBlock, [3, 8, 36, 3], **kwargs)


def IR_200(**kwargs) -> IR:
    """Construct an IR-200 model.
    
    Keyword Args:
        input_size (List[int]): Input image size as [H, W]. Supports [112, 112] 
            or [224, 224]. Defaults to [112, 112].
        num_features (int): Output embedding dimension. Defaults to 512.
        dropout_rate (float): Dropout probability. Defaults to 0.4.
        zero_init_residual (bool): Zero-initialize last BN. Defaults to True.
        
    Returns:
        IR: IR-200 model instance.
    """
    return IR(IRBasicBlock, [6, 26, 60, 6], **kwargs)