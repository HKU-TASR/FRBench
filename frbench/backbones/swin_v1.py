"""Swin Transformer V1 backbone for face recognition.

Reference Paper:
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    (https://arxiv.org/abs/2103.14030)
Reference Code:
    https://github.com/microsoft/Swin-Transformer
    https://github.com/JDAI-CV/FaceX-Zoo/blob/main/backbone/Swin_Transformer.py

Supported Variants: SwinV1-T, SwinV1-S, SwinV1-B, SwinV1-L
"""
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Sequential,
    BatchNorm1d,
    Linear,
    Dropout,
    LayerNorm,
    Identity,
)

from .utils import Flatten, trunc_normal_
from .swin_commons import (
    PatchEmbed,
    PatchMerging,
    BasicLayer,
    _get_swin_config_by_resolution,
)


__all__ = [
    "SwinTransformerV1",
    "SwinV1_Tiny",
    "SwinV1_Small",
    "SwinV1_Base",
    "SwinV1_Large",
]


class SwinTransformerV1(Module):
    """Swin Transformer V1 backbone for face recognition.

    Attributes:
        num_layers: Number of stages.
        embed_dim: Patch embedding dimension.
        num_features: Number of output features.
        ape: Whether to use absolute position embedding.
        patch_norm: Whether to apply normalization after patch embedding.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
    """

    def __init__(
        self,
        input_size: List[int] = [112, 112],
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: type = LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        num_features: int = 512,
        feat_bn: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize SwinTransformerV1.

        Args:
            input_size: Input image size as [H, W]. Defaults to [112, 112].
            patch_size: Patch token size. Defaults to 4.
            in_chans: Number of input image channels. Defaults to 3.
            embed_dim: Patch embedding dimension. Defaults to 96.
            depths: Depth of each Swin Transformer stage. Defaults to [2, 2, 6, 2].
            num_heads: Number of attention heads in each stage. Defaults to [3, 6, 12, 24].
            window_size: Window size. Defaults to 7.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
            qkv_bias: If True, add learnable bias to Q, K, V. Defaults to True.
            qk_scale: Override default QK scale. Defaults to None.
            drop_rate: Dropout rate. Defaults to 0.0.
            attn_drop_rate: Attention dropout rate. Defaults to 0.0.
            drop_path_rate: Stochastic depth rate. Defaults to 0.1.
            norm_layer: Normalization layer class. Defaults to LayerNorm.
            ape: If True, add absolute position embedding. Defaults to False.
            patch_norm: If True, add normalization after patch embedding. Defaults to True.
            use_checkpoint: Whether to use gradient checkpointing. Defaults to False.
            num_features: Output feature dimension. Defaults to 512.
            feat_bn: Whether to apply BatchNorm to output features. Defaults to True.
            dropout_rate: Dropout rate before final feature layer. Defaults to 0.0.
        """
        super(SwinTransformerV1, self).__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            input_size=input_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
        final_h = patches_resolution[0] // (2 ** (self.num_layers - 1))
        final_w = patches_resolution[1] // (2 ** (self.num_layers - 1))
        assert final_h > 0 and final_w > 0, (
            f"Final-stage resolution must be positive, got ({final_h}, {final_w}). "
            f"Check that input_size/patch_size produces enough downsampling stages."
        )
        self.final_resolution = (final_h, final_w)
        flat_dim = final_h * final_w * self.num_features

        # Face recognition output head (FaceXZoo-style, position-preserving):
        self.output_layer = Sequential(
            norm_layer(self.num_features),
            Flatten(),
            Dropout(p=dropout_rate) if dropout_rate > 0 else Identity(),
            Linear(flat_dim, num_features),
            BatchNorm1d(num_features) if feat_bn else Identity(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: Module) -> None:
        """Initialize weights for different layer types.

        Args:
            m: Module to initialize.
        """
        if isinstance(m, Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        """Return parameters that should not have weight decay.

        Returns:
            Set of parameter names.
        """
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> set:
        """Return parameter name keywords that should not have weight decay.

        Returns:
            Set of keywords.
        """
        return {'relative_position_bias_table'}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-token features from the input image.

        Args:
            x: Input image of shape (B, C, H, W).

        Returns:
            Per-token features of shape (B, L, C) where L = final_h * final_w
            and C = self.num_features.
        """
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        return x  # (B, L, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input image of shape (B, C, H, W).

        Returns:
            Face embedding of shape (B, feat_dim).
        """
        x = self.forward_features(x)
        x = self.output_layer(x)
        return x


def SwinV1_Tiny(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> SwinTransformerV1:
    """Create SwinV1-Tiny model for face recognition.

    Args:
        input_size: Input image size as [H, W]. Defaults to [112, 112].
        num_features: Output feature dimension. Defaults to 512.
        dropout_rate: Dropout rate. Defaults to 0.0.
        **kwargs: Additional arguments for SwinTransformerV1.

    Returns:
        SwinV1-Tiny model instance.
    """
    depths, num_heads, patch_size = _get_swin_config_by_resolution(
        input_size,
        depths_4stage=[2, 2, 6, 2],
        num_heads_4stage=[3, 6, 12, 24],
    )
    return SwinTransformerV1(
        input_size=input_size,
        patch_size=patch_size,
        embed_dim=96,
        depths=depths,
        num_heads=num_heads,
        window_size=7,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **kwargs,
    )


def SwinV1_Small(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> SwinTransformerV1:
    """Create SwinV1-Small model for face recognition.

    - 224x224: patch_size=4, 4 stages (56 -> 28 -> 14 -> 7)
    - 112x112: patch_size=2, 4 stages (56 -> 28 -> 14 -> 7)

    Args:
        input_size: Input image size as [H, W]. Defaults to [112, 112].
        num_features: Output feature dimension. Defaults to 512.
        dropout_rate: Dropout rate. Defaults to 0.0.
        **kwargs: Additional arguments for SwinTransformerV1.

    Returns:
        SwinV1-Small model instance.
    """
    depths, num_heads, patch_size = _get_swin_config_by_resolution(
        input_size,
        depths_4stage=[2, 2, 18, 2],
        num_heads_4stage=[3, 6, 12, 24],
    )
    return SwinTransformerV1(
        input_size=input_size,
        patch_size=patch_size,
        embed_dim=96,
        depths=depths,
        num_heads=num_heads,
        window_size=7,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **kwargs,
    )


def SwinV1_Base(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> SwinTransformerV1:
    """Create SwinV1-Base model for face recognition.

    Args:
        input_size: Input image size as [H, W]. Defaults to [112, 112].
        num_features: Output feature dimension. Defaults to 512.
        dropout_rate: Dropout rate. Defaults to 0.0.
        **kwargs: Additional arguments for SwinTransformerV1.

    Returns:
        SwinV1-Base model instance.
    """
    depths, num_heads, patch_size = _get_swin_config_by_resolution(
        input_size,
        depths_4stage=[2, 2, 18, 2],
        num_heads_4stage=[4, 8, 16, 32],
    )
    return SwinTransformerV1(
        input_size=input_size,
        patch_size=patch_size,
        embed_dim=128,
        depths=depths,
        num_heads=num_heads,
        window_size=7,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **kwargs,
    )


def SwinV1_Large(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> SwinTransformerV1:
    """Create SwinV1-Large model for face recognition.

    Args:
        input_size: Input image size as [H, W]. Defaults to [112, 112].
        num_features: Output feature dimension. Defaults to 512.
        dropout_rate: Dropout rate. Defaults to 0.0.
        **kwargs: Additional arguments for SwinTransformerV1.

    Returns:
        SwinV1-Large model instance.
    """
    depths, num_heads, patch_size = _get_swin_config_by_resolution(
        input_size,
        depths_4stage=[2, 2, 18, 2],
        num_heads_4stage=[6, 12, 24, 48],
    )
    return SwinTransformerV1(
        input_size=input_size,
        patch_size=patch_size,
        embed_dim=192,
        depths=depths,
        num_heads=num_heads,
        window_size=7,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **kwargs,
    )
