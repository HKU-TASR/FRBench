"""MobileViTv3 backbone for face recognition.

Reference Paper:
    MobileViTv3: Mobile-Friendly Vision Transformer with Simple and Effective
    Fusion of Local, Global and Input Features (https://arxiv.org/abs/2209.15159)
Reference Code: micronDLA/MobileViTv3, jaiwei98/mobile-vit-pytorch.

Supported Variants:
    MobileViTv3-XXS, -XS, -S       (V1-based block)
    MobileViTv3-0.5 ... -2.0       (V2-based block, width multipliers)
"""
from typing import List, Tuple

import torch
from torch.nn import Module, Sequential, LayerNorm, GroupNorm

from .utils import build_fr_head, infer_output_size, init_weights
from .mobilevit_commons import (
    ConvLayer,
    InvertedResidual,
    TransformerEncoder,
    LinearAttnFFN,
    unfolding_interpolate,
    folding_interpolate,
    unfolding_unfold,
    folding_fold,
)


__all__ = [
    "MobileViTv3_V1",
    "MobileViTv3_V2",
    # V3 (V1-based) Variants
    "MobileViTv3_XXS",
    "MobileViTv3_XS",
    "MobileViTv3_S",
    # V3 (V2-based) Variants
    "MobileViTv3_050",
    "MobileViTv3_075",
    "MobileViTv3_100",
    "MobileViTv3_125",
    "MobileViTv3_150",
    "MobileViTv3_175",
    "MobileViTv3_200",
]


# V3 (V1-based) per-variant configuration (micronDLA configs).
_MOBILEVITV3_V1_CFG = {
    "xxs": dict(mv2_exp_mult=2, ffn_multiplier=2, last_layer_exp_factor=4,
                channels=[16, 16, 24, 64, 80, 128], attn_dim=[64, 80, 96]),
    "xs": dict(mv2_exp_mult=4, ffn_multiplier=2, last_layer_exp_factor=4,
               channels=[16, 32, 48, 96, 160, 160], attn_dim=[96, 120, 144]),
    "s": dict(mv2_exp_mult=4, ffn_multiplier=2, last_layer_exp_factor=3,
              channels=[16, 32, 64, 128, 256, 320], attn_dim=[144, 192, 240]),
}


class MobileViTBlockV3_V1(Module):
    """MobileViTv3 block built on the V1 (quadratic-attention) block.

    Applies the four MobileViTv3 changes to the V1 block: depthwise local conv,
    a 1x1 fusion conv, fusing local+global features, and an input residual.

    Attributes:
        local_rep: Depthwise 3x3 conv + 1x1 projection to the transformer dim.
        global_rep: Stack of Transformer encoders + a final LayerNorm.
        conv_proj: 1x1 conv projecting the transformer dim back to C.
        fusion: 1x1 conv fusing the concatenated local and global features.
    """

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_multiplier: int,
        num_heads: int,
        attn_blocks: int,
        patch_size: Tuple[int, int],
    ) -> None:
        """Initialize MobileViTBlockV3_V1.

        Args:
            in_channels: Block input/output channels (C).
            transformer_dim: Transformer token dimension (d).
            ffn_multiplier: FFN expansion factor relative to ``transformer_dim``.
            num_heads: Number of attention heads.
            attn_blocks: Number of stacked Transformer encoders.
            patch_size: ``(patch_h, patch_w)`` for the unfold/fold operation.
        """
        super(MobileViTBlockV3_V1, self).__init__()
        self.patch_h, self.patch_w = patch_size

        self.local_rep = Sequential(
            ConvLayer(in_channels, in_channels, kernel_size=3, groups=in_channels),
            ConvLayer(in_channels, transformer_dim, kernel_size=1, use_norm=False, use_act=False),
        )

        ffn_dim = int((ffn_multiplier * transformer_dim) // 16 * 16)
        global_rep: List[Module] = [
            TransformerEncoder(transformer_dim, ffn_dim, num_heads) for _ in range(attn_blocks)
        ]
        global_rep.append(LayerNorm(transformer_dim, eps=1e-5))
        self.global_rep = Sequential(*global_rep)

        self.conv_proj = ConvLayer(transformer_dim, in_channels, kernel_size=1)
        self.fusion = ConvLayer(in_channels + transformer_dim, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature map of shape (B, C, H, W).

        Returns:
            Output feature map of shape (B, C, H, W).
        """
        res = x
        fm_conv = self.local_rep(x)
        patches, info_dict = unfolding_interpolate(fm_conv, self.patch_h, self.patch_w)
        patches = self.global_rep(patches)
        fm = folding_interpolate(patches, info_dict, self.patch_h, self.patch_w)
        fm = self.conv_proj(fm)
        fm = self.fusion(torch.cat((fm_conv, fm), dim=1))
        fm = fm + res
        return fm


class MobileViTBlockV3_V2(Module):
    """MobileViTv3 block built on the V2 (separable-attention) block.

    Re-introduces a fusion step that V2 had removed, with the MobileViTv3
    changes: the 1x1 ``conv_proj`` fuses the concatenated global + local features
    (changes 1 & 2) and the input is added back as a residual (change 3). The
    local conv is already depthwise in V2, so change 4 does not apply.

    Attributes:
        local_rep: Depthwise 3x3 conv + 1x1 projection to the attention dim.
        global_rep: Stack of separable-attention FFN blocks + a final GroupNorm.
        conv_proj: 1x1 conv fusing the concatenated global + local features (2*d -> C).
    """

    def __init__(
        self,
        in_channels: int,
        attn_dim: int,
        ffn_multiplier: int,
        attn_blocks: int,
        patch_size: Tuple[int, int],
    ) -> None:
        """Initialize MobileViTBlockV3_V2.

        Args:
            in_channels: Block input/output channels (C).
            attn_dim: Attention-unit dimension (d).
            ffn_multiplier: FFN expansion factor relative to ``attn_dim``.
            attn_blocks: Number of stacked separable-attention FFN blocks.
            patch_size: ``(patch_h, patch_w)`` for the unfold/fold operation.
        """
        super(MobileViTBlockV3_V2, self).__init__()
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

        self.conv_proj = ConvLayer(2 * attn_dim, in_channels, kernel_size=1, use_act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature map of shape (B, C, H, W).

        Returns:
            Output feature map of shape (B, C, H, W).
        """
        res = x
        fm_conv = self.local_rep(x)
        patches, info_dict = unfolding_unfold(fm_conv, self.patch_h, self.patch_w)
        patches = self.global_rep(patches)
        fm = folding_fold(patches, info_dict, self.patch_h, self.patch_w)
        fm = self.conv_proj(torch.cat((fm, fm_conv), dim=1))
        fm = fm + res
        return fm


class MobileViTv3_V1(Module):
    """MobileViTv3 (V1-based) backbone for face recognition.

    Attributes:
        conv_0: 3x3 stride-2 stem.
        layer_1..layer_5: MV2 / MobileViTv3 stages.
        conv_1x1_exp: 1x1 channel-expansion conv before the head.
        output_layer: Position-preserving embedding head.
        final_resolution: ``(H, W)`` of the feature map fed to the head.
    """

    def __init__(
        self,
        mode: str = "s",
        input_size: List[int] = [112, 112],
        num_features: int = 512,
        feat_bn: bool = True,
        dropout_rate: float = 0.0,
        patch_size: Tuple[int, int] = (2, 2),
    ) -> None:
        """Initialize MobileViTv3_V1.

        Args:
            mode: Variant key, one of ``{"xxs", "xs", "s"}``. Defaults to "s".
            input_size: Input image size as [H, W]. Defaults to [112, 112].
            num_features: Output embedding dimension. Defaults to 512.
            feat_bn: Whether to append a BatchNorm1d (BN-Neck). Defaults to True.
            dropout_rate: Dropout before the head's linear layer. Defaults to 0.0.
            patch_size: ``(patch_h, patch_w)`` for every MobileViTv3 block.
                Defaults to (2, 2).
        """
        super(MobileViTv3_V1, self).__init__()
        assert mode in _MOBILEVITV3_V1_CFG, (
            f"Unknown MobileViTv3 (V1) mode '{mode}', "
            f"expected one of {list(_MOBILEVITV3_V1_CFG)}."
        )
        cfg = _MOBILEVITV3_V1_CFG[mode]
        channels = cfg["channels"]
        attn_dim = cfg["attn_dim"]
        exp = cfg["mv2_exp_mult"]
        ffn_mult = cfg["ffn_multiplier"]
        last_exp = cfg["last_layer_exp_factor"]

        self.conv_0 = ConvLayer(3, channels[0], kernel_size=3, stride=2)

        self.layer_1 = Sequential(
            InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=exp),
        )
        self.layer_2 = Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=exp),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=exp),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=exp),
        )

        self.layer_3 = Sequential(
            InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=exp),
            MobileViTBlockV3_V1(channels[3], attn_dim[0], ffn_mult, num_heads=4,
                                attn_blocks=2, patch_size=patch_size),
        )
        self.layer_4 = Sequential(
            InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=exp),
            MobileViTBlockV3_V1(channels[4], attn_dim[1], ffn_mult, num_heads=4,
                                attn_blocks=4, patch_size=patch_size),
        )
        self.layer_5 = Sequential(
            InvertedResidual(channels[4], channels[5], stride=2, expand_ratio=exp),
            MobileViTBlockV3_V1(channels[5], attn_dim[2], ffn_mult, num_heads=4,
                                attn_blocks=3, patch_size=patch_size),
        )
        self.conv_1x1_exp = ConvLayer(channels[5], channels[5] * last_exp, kernel_size=1)

        self.final_resolution = infer_output_size(input_size, num_downsamples=5)
        self.output_layer = build_fr_head(
            channels[5] * last_exp, self.final_resolution, num_features, dropout_rate, feat_bn
        )

        self.apply(init_weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the final-stage feature map.

        Args:
            x: Input image of shape (B, 3, H, W).

        Returns:
            Feature map of shape (B, channels[-1]*last_layer_exp_factor, H', W').
        """
        x = self.conv_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
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


class MobileViTv3_V2(Module):
    """MobileViTv3 (V2-based) backbone for face recognition.

    Same macro layout as ``MobileViTv2`` but every MobileViTv2 block is the
    redesigned ``MobileViTBlockV3_V2`` (fusion re-introduced with the V3
    changes). Channel widths scale with ``width_multiplier``. Only the head is
    the FaceX-Zoo position-preserving variant; there is no 1x1 channel-expansion
    before it. Five stride-2 stages reduce a 112x112 crop to a 4x4 grid.

    Attributes:
        conv_0: 3x3 stride-2 stem.
        layer_1..layer_5: MV2 / MobileViTv3 stages.
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
        """Initialize MobileViTv3_V2.

        Args:
            width_multiplier: Channel width scale (one of 0.5, 0.75, 1.0, 1.25,
                1.5, 1.75, 2.0). Defaults to 1.0.
            input_size: Input image size as [H, W]. Defaults to [112, 112].
            num_features: Output embedding dimension. Defaults to 512.
            feat_bn: Whether to append a BatchNorm1d (BN-Neck). Defaults to True.
            dropout_rate: Dropout before the head's linear layer. Defaults to 0.0.
            patch_size: ``(patch_h, patch_w)`` for every MobileViTv3 block.
                Defaults to (2, 2).
        """
        super(MobileViTv3_V2, self).__init__()
        w = width_multiplier

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
        exp = 2

        self.conv_0 = ConvLayer(3, channels[0], kernel_size=3, stride=2)

        self.layer_1 = Sequential(
            InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=exp),
        )
        self.layer_2 = Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=exp),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=exp),
        )

        self.layer_3 = Sequential(
            InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=exp),
            MobileViTBlockV3_V2(channels[3], attn_dim[0], ffn_mult,
                                attn_blocks=2, patch_size=patch_size),
        )
        self.layer_4 = Sequential(
            InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=exp),
            MobileViTBlockV3_V2(channels[4], attn_dim[1], ffn_mult,
                                attn_blocks=4, patch_size=patch_size),
        )
        self.layer_5 = Sequential(
            InvertedResidual(channels[4], channels[5], stride=2, expand_ratio=exp),
            MobileViTBlockV3_V2(channels[5], attn_dim[2], ffn_mult,
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


def MobileViTv3_XXS(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv3_V1:
    """Create a MobileViTv3-XXS model (V1-based, ~1.25M params).."""
    return MobileViTv3_V1(mode="xxs", input_size=input_size, num_features=num_features,
                          dropout_rate=dropout_rate, **kwargs)


def MobileViTv3_XS(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv3_V1:
    """Create a MobileViTv3-XS model (V1-based, ~2.5M params).."""
    return MobileViTv3_V1(mode="xs", input_size=input_size, num_features=num_features,
                          dropout_rate=dropout_rate, **kwargs)


def MobileViTv3_S(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv3_V1:
    """Create a MobileViTv3-S model (V1-based, ~5.8M params).."""
    return MobileViTv3_V1(mode="s", input_size=input_size, num_features=num_features,
                          dropout_rate=dropout_rate, **kwargs)


def _make_mobilevitv3_v2(width_multiplier, input_size, num_features, dropout_rate, **kwargs):
    """Shared factory body for the width-multiplier MobileViTv3 (V2-based) variants."""
    return MobileViTv3_V2(width_multiplier=width_multiplier, input_size=input_size,
                          num_features=num_features, dropout_rate=dropout_rate, **kwargs)


def MobileViTv3_050(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv3_V2:
    """Create a MobileViTv3-0.5 model (V2-based, width multiplier 0.5).."""
    return _make_mobilevitv3_v2(0.5, input_size, num_features, dropout_rate, **kwargs)


def MobileViTv3_075(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv3_V2:
    """Create a MobileViTv3-0.75 model (V2-based, width multiplier 0.75).."""
    return _make_mobilevitv3_v2(0.75, input_size, num_features, dropout_rate, **kwargs)


def MobileViTv3_100(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv3_V2:
    """Create a MobileViTv3-1.0 model (V2-based, width multiplier 1.0).."""
    return _make_mobilevitv3_v2(1.0, input_size, num_features, dropout_rate, **kwargs)


def MobileViTv3_125(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv3_V2:
    """Create a MobileViTv3-1.25 model (V2-based, width multiplier 1.25).."""
    return _make_mobilevitv3_v2(1.25, input_size, num_features, dropout_rate, **kwargs)


def MobileViTv3_150(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv3_V2:
    """Create a MobileViTv3-1.5 model (V2-based, width multiplier 1.5).."""
    return _make_mobilevitv3_v2(1.5, input_size, num_features, dropout_rate, **kwargs)


def MobileViTv3_175(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv3_V2:
    """Create a MobileViTv3-1.75 model (V2-based, width multiplier 1.75).."""
    return _make_mobilevitv3_v2(1.75, input_size, num_features, dropout_rate, **kwargs)


def MobileViTv3_200(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViTv3_V2:
    """Create a MobileViTv3-2.0 model (V2-based, width multiplier 2.0).."""
    return _make_mobilevitv3_v2(2.0, input_size, num_features, dropout_rate, **kwargs)
