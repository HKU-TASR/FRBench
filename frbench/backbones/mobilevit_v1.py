"""MobileViT (V1) backbone for face recognition.

Reference Paper:
    MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer
    (https://arxiv.org/abs/2110.02178)
Reference Code:
    https://github.com/apple/ml-cvnets
    https://github.com/jaiwei98/mobile-vit-pytorch

Supported Variants: MobileViT-XXS, MobileViT-XS, MobileViT-S
"""
from typing import List, Tuple

import torch
from torch.nn import Module, Sequential, LayerNorm

from .utils import build_fr_head, infer_output_size, init_weights
from .mobilevit_commons import (
    ConvLayer,
    InvertedResidual,
    TransformerEncoder,
    unfolding_interpolate,
    folding_interpolate,
)


__all__ = [
    "MobileViT",
    "MobileViT_XXS",
    "MobileViT_XS",
    "MobileViT_S",
]


# Per-variant configuration (official paper Table 4 / cvnets configs).
_MOBILEVIT_CFG = {
    "xxs": dict(mv2_exp_mult=2, ffn_multiplier=2, last_layer_exp_factor=4,
                channels=[16, 16, 24, 48, 64, 80], attn_dim=[64, 80, 96]),
    "xs": dict(mv2_exp_mult=4, ffn_multiplier=2, last_layer_exp_factor=4,
               channels=[16, 32, 48, 64, 80, 96], attn_dim=[96, 120, 144]),
    "s": dict(mv2_exp_mult=4, ffn_multiplier=2, last_layer_exp_factor=4,
              channels=[16, 32, 64, 96, 128, 160], attn_dim=[144, 192, 240]),
}


class MobileViTBlock(Module):
    """MobileViT block: fuse local (conv) and global (transformer) representations.

    Pipeline (arXiv:2110.02178, Fig. 1b): a local n×n conv + 1×1 projection, then
    unfold into non-overlapping patches over which Transformers model global
    dependencies per pixel-position, then fold back, 1×1 project to C, and fuse
    (concat with the input + n×n conv).

    Attributes:
        local_rep: 3x3 conv (local spatial) + 1x1 conv (project to transformer dim).
        global_rep: Stack of Transformer encoders + a final LayerNorm.
        conv_proj: 1x1 conv projecting the transformer dim back to C.
        fusion: 3x3 conv fusing the concatenated input and global features.
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
        """Initialize MobileViTBlock.

        Args:
            in_channels: Block input/output channels (C).
            transformer_dim: Transformer token dimension (d).
            ffn_multiplier: FFN expansion factor relative to ``transformer_dim``.
            num_heads: Number of attention heads.
            attn_blocks: Number of stacked Transformer encoders.
            patch_size: ``(patch_h, patch_w)`` for the unfold/fold operation.
        """
        super(MobileViTBlock, self).__init__()
        self.patch_h, self.patch_w = patch_size

        # Local representation: spatial 3x3 conv followed by a 1x1 projection to d.
        self.local_rep = Sequential(
            ConvLayer(in_channels, in_channels, kernel_size=3),
            ConvLayer(in_channels, transformer_dim, kernel_size=1, use_norm=False, use_act=False),
        )

        # Global representation: Transformers (rounded FFN dim to a multiple of 16).
        ffn_dim = int((ffn_multiplier * transformer_dim) // 16 * 16)
        global_rep: List[Module] = [
            TransformerEncoder(transformer_dim, ffn_dim, num_heads) for _ in range(attn_blocks)
        ]
        global_rep.append(LayerNorm(transformer_dim, eps=1e-5))
        self.global_rep = Sequential(*global_rep)

        # Project back to C, then fuse input + global features with a 3x3 conv.
        self.conv_proj = ConvLayer(transformer_dim, in_channels, kernel_size=1)
        self.fusion = ConvLayer(2 * in_channels, in_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature map of shape (B, C, H, W).

        Returns:
            Output feature map of shape (B, C, H, W).
        """
        res = x
        fm = self.local_rep(x)
        patches, info_dict = unfolding_interpolate(fm, self.patch_h, self.patch_w)
        patches = self.global_rep(patches)
        fm = folding_interpolate(patches, info_dict, self.patch_h, self.patch_w)
        fm = self.conv_proj(fm)
        fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm


class MobileViT(Module):
    """MobileViT (V1) backbone for face recognition.

    Attributes:
        conv_0: 3x3 stride-2 stem.
        layer_1..layer_5: MV2 / MobileViT stages.
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
        """Initialize MobileViT.

        Args:
            mode: Variant key, one of ``{"xxs", "xs", "s"}``. Defaults to "s".
            input_size: Input image size as [H, W]. Defaults to [112, 112].
            num_features: Output embedding dimension. Defaults to 512.
            feat_bn: Whether to append a BatchNorm1d (BN-Neck). Defaults to True.
            dropout_rate: Dropout before the head's linear layer. Defaults to 0.0.
            patch_size: ``(patch_h, patch_w)`` for every MobileViT block.
                Defaults to (2, 2).
        """
        super(MobileViT, self).__init__()
        assert mode in _MOBILEVIT_CFG, (
            f"Unknown MobileViT mode '{mode}', expected one of {list(_MOBILEVIT_CFG)}."
        )
        cfg = _MOBILEVIT_CFG[mode]
        channels = cfg["channels"]
        attn_dim = cfg["attn_dim"]
        exp = cfg["mv2_exp_mult"]
        ffn_mult = cfg["ffn_multiplier"]
        last_exp = cfg["last_layer_exp_factor"]

        # Stem.
        self.conv_0 = ConvLayer(3, channels[0], kernel_size=3, stride=2)

        # Pure MV2 stages.
        self.layer_1 = Sequential(
            InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=exp),
        )
        self.layer_2 = Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=exp),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=exp),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=exp),
        )

        # MV2 downsample + MobileViT block stages.
        self.layer_3 = Sequential(
            InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=exp),
            MobileViTBlock(channels[3], attn_dim[0], ffn_mult, num_heads=4,
                           attn_blocks=2, patch_size=patch_size),
        )
        self.layer_4 = Sequential(
            InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=exp),
            MobileViTBlock(channels[4], attn_dim[1], ffn_mult, num_heads=4,
                           attn_blocks=4, patch_size=patch_size),
        )
        self.layer_5 = Sequential(
            InvertedResidual(channels[4], channels[5], stride=2, expand_ratio=exp),
            MobileViTBlock(channels[5], attn_dim[2], ffn_mult, num_heads=4,
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


def MobileViT_XXS(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViT:
    """Create a MobileViT-XXS model for face recognition (~1.3M params).."""
    return MobileViT(mode="xxs", input_size=input_size, num_features=num_features,
                     dropout_rate=dropout_rate, **kwargs)


def MobileViT_XS(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViT:
    """Create a MobileViT-XS model for face recognition (~2.3M params).."""
    return MobileViT(mode="xs", input_size=input_size, num_features=num_features,
                     dropout_rate=dropout_rate, **kwargs)


def MobileViT_S(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> MobileViT:
    """Create a MobileViT-S model for face recognition (~5.6M params).."""
    return MobileViT(mode="s", input_size=input_size, num_features=num_features,
                     dropout_rate=dropout_rate, **kwargs)
