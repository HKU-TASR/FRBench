"""MobileViTv2 backbone for face recognition.

Reference Paper:
    Separable Self-attention for Mobile Vision Transformers
    (https://arxiv.org/abs/2206.02680)
Reference Code:
    https://github.com/apple/ml-cvnets
    https://github.com/jaiwei98/mobile-vit-pytorch

Supported Variants:
    MobileViTv2-0.5, -0.75, -1.0, -1.25, -1.5, -1.75, -2.0 (width multipliers)
"""
from typing import List, Tuple

import torch
from torch.nn import Module, Sequential, GroupNorm

from .utils import build_fr_head, infer_output_size, init_weights
from .mobilevit_commons import (
    ConvLayer,
    InvertedResidual,
    LinearAttnFFN,
    unfolding_unfold,
    folding_fold,
)


__all__ = [
    "MobileViTv2",
    "MobileViTv2_050",
    "MobileViTv2_075",
    "MobileViTv2_100",
    "MobileViTv2_125",
    "MobileViTv2_150",
    "MobileViTv2_175",
    "MobileViTv2_200",
]


class MobileViTBlockV2(Module):
    """MobileViTv2 block: local conv + linear-attention global representation.

    Differs from the V1 block (arXiv:2206.02680): the local 3x3 conv is
    depthwise, the global representation uses the O(k) separable self-attention,
    and there is no fusion block / skip connection -- the projected global
    features are returned directly.

    Attributes:
        local_rep: Depthwise 3x3 conv + 1x1 projection to the attention dim.
        global_rep: Stack of separable-attention FFN blocks + a final GroupNorm.
        conv_proj: 1x1 conv projecting the attention dim back to C (no activation).
    """

    def __init__(
        self,
        in_channels: int,
        attn_dim: int,
        ffn_multiplier: int,
        attn_blocks: int,
        patch_size: Tuple[int, int],
    ) -> None:
        """Initialize MobileViTBlockV2.

        Args:
            in_channels: Block input/output channels (C).
            attn_dim: Attention-unit dimension (d).
            ffn_multiplier: FFN expansion factor relative to ``attn_dim``.
            attn_blocks: Number of stacked separable-attention FFN blocks.
            patch_size: ``(patch_h, patch_w)`` for the unfold/fold operation.
        """
        super(MobileViTBlockV2, self).__init__()
        self.patch_h, self.patch_w = patch_size

        self.local_rep = Sequential(
            ConvLayer(in_channels, in_channels, kernel_size=3, groups=in_channels),
            ConvLayer(in_channels, attn_dim, kernel_size=1, use_norm=False, use_act=False),
        )

        ffn_dim = int((ffn_multiplier * attn_dim) // 16 * 16)
        global_rep: List[Module] = [
            LinearAttnFFN(attn_dim, ffn_dim) for _ in range(attn_blocks)
        ]
        global_rep.append(GroupNorm(num_groups=1, num_channels=attn_dim, eps=1e-5))
        self.global_rep = Sequential(*global_rep)

        # Project back to C (no fusion / skip in V2).
        self.conv_proj = ConvLayer(attn_dim, in_channels, kernel_size=1, use_act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature map of shape (B, C, H, W).

        Returns:
            Output feature map of shape (B, C, H, W).
        """
        fm = self.local_rep(x)
        patches, info_dict = unfolding_unfold(fm, self.patch_h, self.patch_w)
        patches = self.global_rep(patches)
        fm = folding_fold(patches, info_dict, self.patch_h, self.patch_w)
        fm = self.conv_proj(fm)
        return fm


class MobileViTv2(Module):
    """MobileViTv2 backbone for face recognition.

    Attributes:
        conv_0: 3x3 stride-2 stem.
        layer_1..layer_5: MV2 / MobileViTv2 stages.
        output_layer: Position-preserving embedding head.
        final_resolution: ``(H, W)`` of the feature map fed to the head.
    """

    def __init__(
        self,
        width_multiplier: float = 1.0,
        input_size: List[int] = [112, 112],
        num_features: int = 512,
        feat_bn: bool = True,
        dropout_rate: float = 0.0,
        patch_size: Tuple[int, int] = (2, 2),
    ) -> None:
        """Initialize MobileViTv2.

        Args:
            width_multiplier: Channel width scale (one of 0.5, 0.75, 1.0, 1.25,
                1.5, 1.75, 2.0). Defaults to 1.0.
            input_size: Input image size as [H, W]. Defaults to [112, 112].
            num_features: Output embedding dimension. Defaults to 512.
            feat_bn: Whether to append a BatchNorm1d (BN-Neck). Defaults to True.
            dropout_rate: Dropout before the head's linear layer. Defaults to 0.0.
            patch_size: ``(patch_h, patch_w)`` for every MobileViTv2 block.
                Defaults to (2, 2).
        """
        super(MobileViTv2, self).__init__()
        w = width_multiplier

        # Width-scaled channels and attention dims (cvnets formula).
        channels = [
            int(max(16, min(64, 32 * w))),
            int(64 * w),
            int(128 * w),
            int(256 * w),
            int(384 * w),
            int(512 * w),
        ]
        attn_dim = [int(128 * w), int(192 * w), int(256 * w)]
        ffn_mult = 2
        exp = 2  # MV2 expansion factor (V2 default)

        # Stem.
        self.conv_0 = ConvLayer(3, channels[0], kernel_size=3, stride=2)

        # Pure MV2 stages (V2 uses 2 blocks in layer_2, vs. 3 in V1).
        self.layer_1 = Sequential(
            InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=exp),
        )
        self.layer_2 = Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=exp),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=exp),
        )

        # MV2 downsample + MobileViTv2 block stages.
        self.layer_3 = Sequential(
            InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=exp),
            MobileViTBlockV2(channels[3], attn_dim[0], ffn_mult,
                             attn_blocks=2, patch_size=patch_size),
        )
        self.layer_4 = Sequential(
            InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=exp),
            MobileViTBlockV2(channels[4], attn_dim[1], ffn_mult,
                             attn_blocks=4, patch_size=patch_size),
        )
        self.layer_5 = Sequential(
            InvertedResidual(channels[4], channels[5], stride=2, expand_ratio=exp),
            MobileViTBlockV2(channels[5], attn_dim[2], ffn_mult,
                             attn_blocks=3, patch_size=patch_size),
        )

        self.final_resolution = infer_output_size(input_size, num_downsamples=5)
        self.output_layer = build_fr_head(
            channels[-1], self.final_resolution, num_features, dropout_rate, feat_bn
        )

        self.apply(init_weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the final-stage feature map.

        Args:
            x: Input image of shape (B, 3, H, W).

        Returns:
            Feature map of shape (B, channels[-1], H', W').
        """
        x = self.conv_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embeddings from input images.

        Args:
            x: Input images of shape (B, 3, H, W).

        Returns:
            Face embeddings of shape (B, num_features).
        """
        x = self.forward_features(x)
        x = self.output_layer(x)
        return x


def _make_mobilevitv2(width_multiplier, input_size, num_features, dropout_rate, **kwargs):
    """Shared factory body for the width-multiplier MobileViTv2 variants."""
    return MobileViTv2(width_multiplier=width_multiplier, input_size=input_size,
                       num_features=num_features, dropout_rate=dropout_rate, **kwargs)


def MobileViTv2_050(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv2:
    """Create a MobileViTv2-0.5 model (width multiplier 0.5, ~1.4M params).."""
    return _make_mobilevitv2(0.5, input_size, num_features, dropout_rate, **kwargs)


def MobileViTv2_075(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv2:
    """Create a MobileViTv2-0.75 model (width multiplier 0.75).."""
    return _make_mobilevitv2(0.75, input_size, num_features, dropout_rate, **kwargs)


def MobileViTv2_100(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv2:
    """Create a MobileViTv2-1.0 model (width multiplier 1.0, ~4.9M params).."""
    return _make_mobilevitv2(1.0, input_size, num_features, dropout_rate, **kwargs)


def MobileViTv2_125(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv2:
    """Create a MobileViTv2-1.25 model (width multiplier 1.25).."""
    return _make_mobilevitv2(1.25, input_size, num_features, dropout_rate, **kwargs)


def MobileViTv2_150(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv2:
    """Create a MobileViTv2-1.5 model (width multiplier 1.5).."""
    return _make_mobilevitv2(1.5, input_size, num_features, dropout_rate, **kwargs)


def MobileViTv2_175(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv2:
    """Create a MobileViTv2-1.75 model (width multiplier 1.75).."""
    return _make_mobilevitv2(1.75, input_size, num_features, dropout_rate, **kwargs)


def MobileViTv2_200(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv2:
    """Create a MobileViTv2-2.0 model (width multiplier 2.0, ~18.5M params).."""
    return _make_mobilevitv2(2.0, input_size, num_features, dropout_rate, **kwargs)
