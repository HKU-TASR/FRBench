"""Shared components for Swin Transformer variants (V1, V2, SwinFace, SwinMLP).

This module collects the Swin-specific building blocks that are reused across
two or more Swin variants, including:

    - Pure helpers: window_partition, window_reverse, _get_swin_config_by_resolution
    - Modules: PatchEmbed (used by all variants), WindowAttention,
      SwinTransformerBlock, PatchMerging, BasicLayer (used by SwinV1, SwinFace
      and SwinMLP).

Version-specific components live in their corresponding files:
    - swin_v1.py: SwinTransformerV1 + factories
    - swin_v2.py: SwinTransformerV2 + V2-specific blocks + factories
    - swin_face.py: SwinFace + factories
    - swin_mlp.py: SwinMLP + MLP-specific blocks + factories
"""
from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout,
    GELU,
    LayerNorm,
    Identity,
)

from .utils import DropPath, trunc_normal_


__all__ = [
    "to_2tuple",
    "Mlp",
    "window_partition",
    "window_reverse",
    "PatchEmbed",
    "WindowAttention",
    "SwinTransformerBlock",
    "PatchMerging",
    "BasicLayer",
    "_get_swin_config_by_resolution",
]


def to_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert input to a 2-tuple.

    Args:
        x: Integer or tuple of two integers.

    Returns:
        Tuple of two integers.
    """
    if isinstance(x, tuple):
        return x
    return (x, x)


class Mlp(Module):
    """Multi-layer Perceptron (MLP) module.

    A standard 2-layer MLP with GELU activation and dropout.
    Commonly used in Vision Transformers.

    Attributes:
        fc1: First linear layer.
        act: Activation function (GELU by default).
        fc2: Second linear layer.
        drop: Dropout layer.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type = GELU,
        drop: float = 0.0,
    ) -> None:
        """Initialize MLP module.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features. Defaults to in_features.
            out_features: Number of output features. Defaults to in_features.
            act_layer: Activation layer class. Defaults to GELU.
            drop: Dropout probability. Defaults to 0.0.
        """
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows.

    Args:
        x: Input tensor of shape (B, H, W, C).
        window_size: Size of each window.

    Returns:
        Windows tensor of shape (num_windows*B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition to reconstruct feature map.

    Args:
        windows: Windows tensor of shape (num_windows*B, window_size, window_size, C).
        window_size: Size of each window.
        H: Height of the feature map.
        W: Width of the feature map.

    Returns:
        Reconstructed tensor of shape (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(Module):
    """Image to Patch Embedding.

    Converts input image to patch tokens using a single convolution.

    Attributes:
        img_size: Input image size.
        patch_size: Patch token size.
        patches_resolution: Resolution of patch grid.
        num_patches: Total number of patches.
        in_chans: Number of input image channels.
        embed_dim: Embedding dimension.
    """

    def __init__(
        self,
        input_size: List[int] = [224, 224],
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[type] = None,
    ) -> None:
        """Initialize PatchEmbed module.

        Args:
            input_size: Input image size as [H, W]. Defaults to [224, 224].
            patch_size: Patch token size. Defaults to 4.
            in_chans: Number of input image channels. Defaults to 3.
            embed_dim: Embedding dimension. Defaults to 96.
            norm_layer: Normalization layer class. Defaults to None.
        """
        super(PatchEmbed, self).__init__()
        patch_size = to_2tuple(patch_size)
        patches_resolution = [input_size[0] // patch_size[0], input_size[1] // patch_size[1]]
        self.img_size = input_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through patch embedding.

        Args:
            x: Input image of shape (B, C, H, W).

        Returns:
            Patch tokens of shape (B, num_patches, embed_dim).
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class WindowAttention(Module):
    """Window-based Multi-head Self Attention (W-MSA) module with relative position bias.

    Supports both shifted and non-shifted window attention as described in the Swin paper.

    Attributes:
        dim: Number of input channels.
        window_size: Window size (height, width).
        num_heads: Number of attention heads.
        scale: Scaling factor for attention scores.
        relative_position_bias_table: Learnable relative position bias table.
        relative_position_index: Precomputed relative position indices.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Initialize WindowAttention module.

        Args:
            dim: Number of input channels.
            window_size: Window size (height, width).
            num_heads: Number of attention heads.
            qkv_bias: If True, add learnable bias to Q, K, V projections. Defaults to True.
            qk_scale: Override default QK scale (head_dim ** -0.5). Defaults to None.
            attn_drop: Attention dropout probability. Defaults to 0.0.
            proj_drop: Output projection dropout probability. Defaults to 0.0.
        """
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through window attention.

        Args:
            x: Input features of shape (num_windows*B, N, C).
            mask: Attention mask of shape (num_windows, Wh*Ww, Wh*Ww) or None.

        Returns:
            Output features of shape (num_windows*B, N, C).
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )  # Wh*Ww, Wh*Ww, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinTransformerBlock(Module):
    """Swin Transformer Block.

    A single block of the Swin Transformer consisting of window-based multi-head
    self-attention (W-MSA or SW-MSA) followed by a feed-forward network (MLP).

    Attributes:
        dim: Number of input channels.
        input_resolution: Input resolution (height, width).
        num_heads: Number of attention heads.
        window_size: Window size.
        shift_size: Shift size for SW-MSA.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = GELU,
        norm_layer: type = LayerNorm,
    ) -> None:
        """Initialize SwinTransformerBlock.

        Args:
            dim: Number of input channels.
            input_resolution: Input resolution (height, width).
            num_heads: Number of attention heads.
            window_size: Window size. Defaults to 7.
            shift_size: Shift size for SW-MSA. Defaults to 0.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
            qkv_bias: If True, add learnable bias to Q, K, V. Defaults to True.
            qk_scale: Override default QK scale. Defaults to None.
            drop: Dropout probability. Defaults to 0.0.
            attn_drop: Attention dropout probability. Defaults to 0.0.
            drop_path: Stochastic depth rate. Defaults to 0.0.
            act_layer: Activation layer class. Defaults to GELU.
            norm_layer: Normalization layer class. Defaults to LayerNorm.
        """
        super(SwinTransformerBlock, self).__init__()
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

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
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

        # Calculate attention mask for SW-MSA
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1, H, W, 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Swin Transformer block.

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

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B, H, W, C

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
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


class PatchMerging(Module):
    """Patch Merging Layer.

    Reduces spatial resolution by 2x while doubling channels.

    Attributes:
        input_resolution: Input resolution (height, width).
        dim: Number of input channels.
        reduction: Linear layer for channel reduction.
        norm: Normalization layer.
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        norm_layer: type = LayerNorm,
    ) -> None:
        """Initialize PatchMerging layer.

        Args:
            input_resolution: Input resolution (height, width).
            dim: Number of input channels.
            norm_layer: Normalization layer class. Defaults to LayerNorm.
        """
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through patch merging.

        Args:
            x: Input features of shape (B, H*W, C).

        Returns:
            Output features of shape (B, H/2*W/2, 2*C).
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature has wrong size, expected {H * W}, got {L}"
        assert H % 2 == 0 and W % 2 == 0, f"Resolution ({H}*{W}) not divisible by 2"

        x = x.view(B, H, W, C)

        # Concatenate 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, :]  # B, H/2, W/2, C
        x2 = x[:, 0::2, 1::2, :]  # B, H/2, W/2, C
        x3 = x[:, 1::2, 1::2, :]  # B, H/2, W/2, C
        x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C
        x = x.view(B, -1, 4 * C)  # B, H/2*W/2, 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(Module):
    """A basic Swin Transformer layer for one stage.

    Contains multiple Swin Transformer blocks and an optional downsample layer.

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
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        norm_layer: type = LayerNorm,
        downsample: Optional[type] = None,
        use_checkpoint: bool = False,
    ) -> None:
        """Initialize BasicLayer.

        Args:
            dim: Number of input channels.
            input_resolution: Input resolution (height, width).
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
            qkv_bias: If True, add learnable bias to Q, K, V. Defaults to True.
            qk_scale: Override default QK scale. Defaults to None.
            drop: Dropout probability. Defaults to 0.0.
            attn_drop: Attention dropout probability. Defaults to 0.0.
            drop_path: Stochastic depth rate. Defaults to 0.0.
            norm_layer: Normalization layer class. Defaults to LayerNorm.
            downsample: Downsample layer at the end. Defaults to None.
            use_checkpoint: Whether to use gradient checkpointing. Defaults to False.
        """
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through basic layer.

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


def _get_swin_config_by_resolution(
    input_size: List[int],
    depths_4stage: List[int],
    num_heads_4stage: List[int],
) -> Tuple[List[int], List[int], int]:
    """Get depths, num_heads, and patch_size configuration based on input resolution.

    Always preserves the full 4-stage architecture by adapting patch_size:
    For 224x224 input: patch_size=4, 4 stages
        224 -> patch_size=4 -> 56x56 -> 28x28 -> 14x14 -> 7x7 (final)
    For 112x112 input: patch_size=2, 4 stages
        112 -> patch_size=2 -> 56x56 -> 28x28 -> 14x14 -> 7x7 (final)

    Args:
        input_size: Input image size as [H, W] (e.g., [112, 112] or [224, 224]).
        depths_4stage: Depths for 4-stage configuration.
        num_heads_4stage: Num heads for 4-stage configuration.

    Returns:
        Tuple of (depths, num_heads, patch_size) adapted for the given resolution.
    """
    if input_size[0] >= 224:
        return depths_4stage, num_heads_4stage, 4
    else:
        return depths_4stage, num_heads_4stage, 2
