"""SwinMLP: Spatial MLP replacing Window Attention.

No independent paper published. This is an experimental model released by Microsoft

Reference: https://github.com/microsoft/Swin-Transformer (README.md Updates section)

Supported Variants: SwinMLP-T, SwinMLP-S, SwinMLP-B, SwinMLP-L
"""
from typing import Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm1d,
    Linear,
    Dropout,
    GELU,
    LayerNorm,
    Identity,
)

from .utils import DropPath, Flatten, trunc_normal_
from .swin_commons import (
    to_2tuple,
    Mlp,
    PatchEmbed,
    PatchMerging,
    window_partition,
    window_reverse,
    _get_swin_config_by_resolution,
)


__all__ = [
    "SwinMLP",
    "SwinMLP_Tiny",
    "SwinMLP_Small",
    "SwinMLP_Base",
    "SwinMLP_Large",
]


class SwinMLPBlock(Module):
    """Swin MLP Block.

    Attributes:
        dim: Number of input channels.
        input_resolution: Input resolution (height, width).
        num_heads: Number of heads for spatial MLP (controls groups).
        window_size: Window size.
        shift_size: Shift size for Shifted Window MLP.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        padding: Padding values for shift operation [P_l, P_r, P_t, P_b].
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = GELU,
        norm_layer: type = LayerNorm,
    ) -> None:
        """Initialize SwinMLPBlock.

        Args:
            dim: Number of input channels.
            input_resolution: Input resolution (height, width).
            num_heads: Number of heads for grouped spatial MLP.
            window_size: Window size. Defaults to 7.
            shift_size: Shift size for SW-MLP. Defaults to 0.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
            drop: Dropout probability. Defaults to 0.0.
            drop_path: Stochastic depth rate. Defaults to 0.0.
            act_layer: Activation layer class. Defaults to GELU.
            norm_layer: Normalization layer class. Defaults to LayerNorm.
        """
        super(SwinMLPBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # If window size is larger than input resolution, no window partition
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.padding = [
            self.window_size - self.shift_size,  # P_l (left)
            self.shift_size,                      # P_r (right)
            self.window_size - self.shift_size,  # P_t (top)
            self.shift_size,                      # P_b (bottom)
        ]

        self.norm1 = norm_layer(dim)

        self.spatial_mlp = nn.Conv1d(
            self.num_heads * self.window_size ** 2,
            self.num_heads * self.window_size ** 2,
            kernel_size=1,
            groups=self.num_heads,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Swin MLP block.

        Args:
            x: Input features of shape (B, L, C) where L = H * W.

        Returns:
            Output features of shape (B, L, C).
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature has wrong size, expected {H * W}, got {L}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Shift: use padding instead of cyclic shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], mode="constant", value=0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape

        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size^2, C

        x_windows_heads = x_windows.view(
            -1, self.window_size * self.window_size, self.num_heads, C // self.num_heads
        )
        x_windows_heads = x_windows_heads.transpose(1, 2)  # nW*B, num_heads, window_size^2, C//num_heads
        x_windows_heads = x_windows_heads.reshape(
            -1, self.num_heads * self.window_size * self.window_size, C // self.num_heads
        )

        # Apply spatial MLP (Conv1d with groups)
        spatial_mlp_windows = self.spatial_mlp(x_windows_heads)  # nW*B, num_heads*window_size^2, C//num_heads

        spatial_mlp_windows = spatial_mlp_windows.view(
            -1, self.num_heads, self.window_size * self.window_size, C // self.num_heads
        ).transpose(1, 2)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size, C)

        # Merge windows
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(spatial_mlp_windows, self.window_size, _H, _W)  # B, _H, _W, C

        # Reverse shift: remove padding
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # Residual connection
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, "
            f"num_heads={self.num_heads}, window_size={self.window_size}, "
            f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )


class BasicLayerMLP(Module):
    """A basic Swin MLP layer

    Attributes:
        dim: Number of input channels.
        input_resolution: Input resolution (height, width).
        depth: Number of blocks.
        use_checkpoint: Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        norm_layer: type = LayerNorm,
        downsample=None,
        use_checkpoint: bool = False,
    ) -> None:
        """Initialize BasicLayerMLP.

        Args:
            dim: Number of input channels.
            input_resolution: Input resolution (height, width).
            depth: Number of blocks.
            num_heads: Number of heads for spatial MLP.
            window_size: Local window size.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
            drop: Dropout probability. Defaults to 0.0.
            drop_path: Stochastic depth rate. Defaults to 0.0.
            norm_layer: Normalization layer class. Defaults to LayerNorm.
            downsample: Downsample layer at the end. Defaults to None.
            use_checkpoint: Whether to use gradient checkpointing. Defaults to False.
        """
        super(BasicLayerMLP, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build MLP blocks
        self.blocks = nn.ModuleList([
            SwinMLPBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        # Patch merging layer (same as SwinV1)
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through basic MLP layer.

        Args:
            x: Input features of shape (B, L, C).

        Returns:
            Output features of shape (B, L', C') where L' and C' depend on downsample.
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class SwinMLP(Module):
    """Swin MLP backbone for face recognition.

    Key features:
        1. Spatial MLP instead of Window Attention for token mixing
        2. Padding-based shift instead of cyclic shift (no attention mask needed)
        3. Same hierarchical structure as Swin Transformer
        4. Efficient: O(N) complexity vs O(N²) for attention

    For face recognition with 112x112 input:
        - Uses patch_size=2 to preserve all 4 stages
        - 112/2 = 56 initial patches -> 56x56 -> 28x28 -> 14x14 -> 7x7

    Attributes:
        num_classes: Number of classes (kept for compatibility).
        num_layers: Number of transformer stages.
        embed_dim: Embedding dimension.
        num_features: Number of features in final layer.
        patches_resolution: Resolution of patch grid.
    """

    def __init__(
        self,
        input_size: List[int] = [112, 112],
        patch_size: int = 4,
        in_chans: int = 3,
        num_features: int = 512,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: type = LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        feat_bn: bool = True,
        dropout_rate: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize SwinMLP backbone.

        Args:
            input_size: Input image size as [H, W]. Defaults to [112, 112].
            patch_size: Patch size. Defaults to 4.
            in_chans: Number of input image channels. Defaults to 3.
            num_features: Output feature dimension. Defaults to 512.
            embed_dim: Patch embedding dimension. Defaults to 96.
            depths: Depth of each Swin MLP layer. Defaults to [2, 2, 6, 2].
            num_heads: Number of heads in each layer. Defaults to [3, 6, 12, 24].
            window_size: Window size. Defaults to 7.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
            drop_rate: Dropout rate (token / MLP). Defaults to 0.0.
            drop_path_rate: Stochastic depth rate. Defaults to 0.1.
            norm_layer: Normalization layer class. Defaults to LayerNorm.
            ape: If True, add absolute position embedding. Defaults to False.
            patch_norm: If True, add normalization after patch embedding. Defaults to True.
            use_checkpoint: Whether to use gradient checkpointing. Defaults to False.
            feat_bn: Whether to apply BatchNorm to output features (BN-Neck).
                Defaults to True.
            dropout_rate: Dropout rate inside the FR output head, applied between
                the final LayerNorm/Flatten and the embedding Linear. Defaults to 0.0.
            **kwargs: Additional arguments.
        """
        super(SwinMLP, self).__init__()

        self.patch_norm = patch_norm
        self.embed_dim = embed_dim
        self.ape = ape
        self.mlp_ratio = mlp_ratio

        # Compute patches resolution
        patch_size = to_2tuple(patch_size)
        patches_resolution = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])
        self.patches_resolution = patches_resolution

        # Use all provided stages (factory functions handle resolution adaptation via patch_size)
        num_layers = len(depths)
        self.num_layers = num_layers

        self.num_features = int(embed_dim * 2 ** (num_layers - 1))

        # Patch embedding
        self.patch_embed = PatchEmbed(
            input_size=input_size,
            patch_size=patch_size[0],
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build MLP layers
        self.layers = nn.ModuleList()
        for i_layer in range(num_layers):
            layer = BasicLayerMLP(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        final_h = patches_resolution[0] // (2 ** (num_layers - 1))
        final_w = patches_resolution[1] // (2 ** (num_layers - 1))
        assert final_h > 0 and final_w > 0, (
            f"Final-stage resolution must be positive, got ({final_h}, {final_w}). "
            f"Check that input_size/patch_size produces enough downsampling stages."
        )
        self.final_resolution = (final_h, final_w)
        flat_dim = final_h * final_w * self.num_features

        self.output_layer = Sequential(
            norm_layer(self.num_features),
            Flatten(),
            Dropout(p=dropout_rate) if dropout_rate > 0 else Identity(),
            Linear(flat_dim, num_features),
            BatchNorm1d(num_features) if feat_bn else Identity(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: Module) -> None:
        """Initialize weights for the model.

        Args:
            m: Module to initialize.
        """
        if isinstance(m, Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, Conv2d)):
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
            Face recognition embedding of shape (B, feat_dim).
        """
        x = self.forward_features(x)
        x = self.output_layer(x)
        return x


def SwinMLP_Tiny(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> SwinMLP:
    """Create SwinMLP-Tiny model for face recognition.

    Args:
        input_size: Input image size as [H, W]. Defaults to [112, 112].
        num_features: Output embedding dimension. Defaults to 512.
        **kwargs: Additional arguments for SwinMLP.

    Returns:
        SwinMLP-Tiny model instance.
    """
    depths, num_heads, patch_size = _get_swin_config_by_resolution(
        input_size,
        depths_4stage=[2, 2, 6, 2],
        num_heads_4stage=[3, 6, 12, 24],
    )
    return SwinMLP(
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


def SwinMLP_Small(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> SwinMLP:
    """Create SwinMLP-Small model for face recognition.

    Args:
        input_size: Input image size as [H, W]. Defaults to [112, 112].
        num_features: Output embedding dimension. Defaults to 512.
        **kwargs: Additional arguments for SwinMLP.

    Returns:
        SwinMLP-Small model instance.
    """
    depths, num_heads, patch_size = _get_swin_config_by_resolution(
        input_size,
        depths_4stage=[2, 2, 18, 2],
        num_heads_4stage=[3, 6, 12, 24],
    )
    return SwinMLP(
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


def SwinMLP_Base(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> SwinMLP:
    """Create SwinMLP-Base model for face recognition.

    Args:
        input_size: Input image size as [H, W]. Defaults to [112, 112].
        num_features: Output embedding dimension. Defaults to 512.
        **kwargs: Additional arguments for SwinMLP.

    Returns:
        SwinMLP-Base model instance.
    """
    depths, num_heads, patch_size = _get_swin_config_by_resolution(
        input_size,
        depths_4stage=[2, 2, 18, 2],
        num_heads_4stage=[4, 8, 16, 32],
    )
    return SwinMLP(
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


def SwinMLP_Large(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> SwinMLP:
    """Create SwinMLP-Large model for face recognition.

    Args:
        input_size: Input image size as [H, W]. Defaults to [112, 112].
        num_features: Output embedding dimension. Defaults to 512.
        **kwargs: Additional arguments for SwinMLP.

    Returns:
        SwinMLP-Large model instance.
    """
    depths, num_heads, patch_size = _get_swin_config_by_resolution(
        input_size,
        depths_4stage=[2, 2, 18, 2],
        num_heads_4stage=[6, 12, 24, 48],
    )
    return SwinMLP(
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
