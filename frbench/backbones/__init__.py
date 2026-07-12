from typing import Dict, Any
import copy

import torch
from torch import nn

from .resnet import *
from .resnetv2 import *
from .ir import *
from .irse import *
from .densenet import *
from .efficientnet import *
from .mobilefacenet import *
from .mobilenet import *
from .convnext import *
from .swin_v1 import *
from .swin_v2 import *
from .swin_mlp import *
from .mobilevit_v1 import *
from .mobilevit_v2 import *
from .mobilevit_v3 import *

BACKBONE_REGISTRY = {
    # ResNet Variants
    'resnet-18': ResNet_18, 
    'resnet-34': ResNet_34,
    'resnet-50': ResNet_50,
    'resnet-100': ResNet_100,
    'resnet-101': ResNet_101,
    'resnet-152': ResNet_152,
    'resnet-200': ResNet_200,
    # ResNetV2 Variants
    'resnetv2-18': ResNetV2_18,
    'resnetv2-34': ResNetV2_34,
    'resnetv2-50': ResNetV2_50,
    'resnetv2-100': ResNetV2_100,
    'resnetv2-101': ResNetV2_101,
    'resnetv2-152': ResNetV2_152,
    'resnetv2-200': ResNetV2_200,
    # IR Variants
    'ir-18': IR_18,
    'ir-34': IR_34,
    'ir-50': IR_50,
    'ir-100': IR_100,
    'ir-101': IR_101,
    'ir-152': IR_152,
    'ir-200': IR_200,
    # IR-SE Variants
    'irse-18': IR_SE_18,
    'irse-34': IR_SE_34,
    'irse-50': IR_SE_50,
    'irse-100': IR_SE_100,
    'irse-101': IR_SE_101,
    'irse-152': IR_SE_152,
    'irse-185': IR_SE_185,
    'irse-200': IR_SE_200,
    # DenseNet Variants
    'densenet-121': DenseNet_121,
    'densenet-169': DenseNet_169,
    'densenet-201': DenseNet_201,
    # EfficientNetV1 Variants
    'efficientnetv1-b0': EfficientNetV1_B0,
    'efficientnetv1-b1': EfficientNetV1_B1,
    'efficientnetv1-b2': EfficientNetV1_B2,
    'efficientnetv1-b3': EfficientNetV1_B3,
    'efficientnetv1-b4': EfficientNetV1_B4,
    'efficientnetv1-b5': EfficientNetV1_B5,
    'efficientnetv1-b6': EfficientNetV1_B6,
    'efficientnetv1-b7': EfficientNetV1_B7,
    # MobileNet Variants
    'mobilenet-w1': MobileNet_W1,
    'mobilenet-w3d4': MobileNet_W3D4,
    'mobilenet-wd2': MobileNet_WD2,
    'mobilenet-wd4': MobileNet_WD4,
    # MobileNetV2 Variants
    'mobilenetv2-w1': MobileNetV2_W1,
    'mobilenetv2-w3d4': MobileNetV2_W3D4,
    'mobilenetv2-wd2': MobileNetV2_WD2,
    'mobilenetv2-wd4': MobileNetV2_WD4,
    # MobileNetV3 Variants
    'mobilenetv3-l': MobileNetV3_Large,
    'mobilenetv3-s': MobileNetV3_Small,
    # MobileNetV4 Variants
    'mobilenetv4conv-s': MobileNetV4_Conv_Small,
    'mobilenetv4conv-m': MobileNetV4_Conv_Medium,
    'mobilenetv4conv-l': MobileNetV4_Conv_Large,
    'mobilenetv4hybrid-m': MobileNetV4_Hybrid_Medium,
    'mobilenetv4hybrid-l': MobileNetV4_Hybrid_Large,
    # MobileFaceNet Variants
    'mobilefacenet': MobileFaceNet_ECA,
    'mobilefacenet-plain': MobileFaceNet_Plain,
    'convnext-t': ConvNeXt_Tiny,
    'convnext-s': ConvNeXt_Small,
    'convnext-b': ConvNeXt_Base,
    'convnext-l': ConvNeXt_Large,
    'convnext-xl': ConvNeXt_XLarge,
    'convnextv2-atto': ConvNeXtV2_Atto,
    'convnextv2-femto': ConvNeXtV2_Femto,
    'convnextv2-pico': ConvNeXtV2_Pico,
    'convnextv2-n': ConvNeXtV2_Nano,
    'convnextv2-t': ConvNeXtV2_Tiny,
    'convnextv2-s': ConvNeXtV2_Small,
    'convnextv2-b': ConvNeXtV2_Base,
    'convnextv2-l': ConvNeXtV2_Large,
    'convnextv2-xl': ConvNeXtV2_Huge,
    # SwinV1 Variants
    'swinv1-t': SwinV1_Tiny,  # 224 recommended; 112 works with 3 stages instead of 4
    'swinv1-s': SwinV1_Small,
    'swinv1-b': SwinV1_Base,
    'swinv1-l': SwinV1_Large,
    # SwinMLP Variants (experimental spatial MLP; microsoft/Swin-Transformer)
    'swinmlp-t': SwinMLP_Tiny,
    'swinmlp-s': SwinMLP_Small,
    'swinmlp-b': SwinMLP_Base,
    'swinmlp-l': SwinMLP_Large,
    # SwinV2 Variants
    'swinv2-t': SwinV2_Tiny,  # 224 recommended; 112 works with 3 stages instead of 4
    'swinv2-s': SwinV2_Small,
    'swinv2-b': SwinV2_Base,
    'swinv2-l': SwinV2_Large,
    # MobileViT (V1) Variants
    'mobilevit-xxs': MobileViT_XXS,
    'mobilevit-xs': MobileViT_XS,
    'mobilevit-s': MobileViT_S,
    # MobileViTv2 Variants (separable self-attention, width multipliers)
    'mobilevitv2-0.5': MobileViTv2_050,
    'mobilevitv2-0.75': MobileViTv2_075,
    'mobilevitv2-1.0': MobileViTv2_100,
    'mobilevitv2-1.25': MobileViTv2_125,
    'mobilevitv2-1.5': MobileViTv2_150,
    'mobilevitv2-1.75': MobileViTv2_175,
    'mobilevitv2-2.0': MobileViTv2_200,
    # MobileViTv3 (V1-based) Variants
    'mobilevitv3-xxs': MobileViTv3_XXS,
    'mobilevitv3-xs': MobileViTv3_XS,
    'mobilevitv3-s': MobileViTv3_S,
    # MobileViTv3 (V2-based) Variants (width multipliers)
    'mobilevitv3-0.5': MobileViTv3_050,
    'mobilevitv3-0.75': MobileViTv3_075,
    'mobilevitv3-1.0': MobileViTv3_100,
    'mobilevitv3-1.25': MobileViTv3_125,
    'mobilevitv3-1.5': MobileViTv3_150,
    'mobilevitv3-1.75': MobileViTv3_175,
    'mobilevitv3-2.0': MobileViTv3_200,
}

def build_backbone(backbone_name, backbone_kwargs: Dict[str, Any], device: torch.device) -> nn.Module:
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Backbone '{backbone_name}' is not registered. "
            f"Available backbones: {list(BACKBONE_REGISTRY.keys())}"
        )
    backbone_kwargs = copy.deepcopy(backbone_kwargs)
    return BACKBONE_REGISTRY[backbone_name](**backbone_kwargs).to(device)