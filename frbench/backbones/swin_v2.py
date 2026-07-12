"""Swin Transformer V2 backbone for face recognition.

Reference Paper:
    Swin Transformer V2: Scaling Up Capacity and Resolution
    (https://arxiv.org/abs/2111.09883)
Reference Code:
    https://github.com/microsoft/Swin-Transformer

Supported Variants: SwinV2-T, SwinV2-S, SwinV2-B, SwinV2-L
"""
import math
from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import (
    Module,
    Sequential,
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
    window_partition,
    window_reverse,
    _get_swin_config_by_resolution,
)


__all__ = [
    "SwinTransformerV2",
    "SwinV2_Tiny",
    "SwinV2_Small",
    "SwinV2_Base",
    "SwinV2_Large",
]


class WindowAttentionV2(Module):
    """Window-based Multi-head Self Attention (W-MSA) module for Swin Transformer V2.

    Attributes:
        dim: Number of input channels.
        window_size: Window size (height, width).
        pretrained_window_size: Window size used in pre-training.
        num_heads: Number of attention heads.
        logit_scale: Learnable logit scale for cosine attention.
        cpb_mlp: MLP for continuous position bias generation.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pretrained_window_size: Tuple[int, int] = (0, 0),
    ) -> None:
        """Initialize WindowAttentionV2 module.

        Args:
            dim: Number of input channels.
            window_size: Window size (height, width).
            num_heads: Number of attention heads.
            qkv_bias: If True, add learnable bias to Q and V projections. Defaults to True.
            attn_drop: Attention dropout probability. Defaults to 0.0.
            proj_drop: Output projection dropout probability. Defaults to 0.0.
            pretrained_window_size: Window size in pre-training. Defaults to (0, 0).
        """
        super(WindowAttentionV2, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        # Learnable logit scale for cosine attention (per head)
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        # MLP to generate continuous relative position bias
        self.cpb_mlp = Sequential(
            Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            Linear(512, num_heads, bias=False),
        )

        # Build log-spaced relative coordinates table
        # Shape: 1, 2*Wh-1, 2*Ww-1, 2
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
        )
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        # Normalize coordinates
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

        # Apply log-spacing: normalize to [-8, 8], then log2
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table) *
            torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        )

        self.register_buffer("relative_coords_table", relative_coords_table)

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

        # QKV projection - no bias on K (only Q and V have bias)
        self.qkv = Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through window attention V2.

        Args:
            x: Input features of shape (num_windows*B, N, C).
            mask: Attention mask of shape (num_windows, Wh*Ww, Wh*Ww) or None.

        Returns:
            Output features of shape (num_windows*B, N, C).
        """
        B_, N, C = x.shape

        # Construct QKV bias (no bias on K)
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)
            )

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Make torchscript happy

        # Cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))

        # Apply learnable logit scale (clamped to max temperature of 100)
        logit_scale = torch.clamp(
            self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01, device=self.logit_scale.device))
        ).exp()
        attn = attn * logit_scale

        # Generate continuous position bias via MLP
        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table
        ).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )  # Wh*Ww, Wh*Ww, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, Wh*Ww, Wh*Ww

        # Scale bias to [0, 16] range via sigmoid
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
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
        return (
            f'dim={self.dim}, window_size={self.window_size}, '
            f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'
        )


class SwinTransformerBlockV2(Module):
    """Swin Transformer V2 Block with post-norm and cosine attention.

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
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = GELU,
        norm_layer: type = LayerNorm,
        pretrained_window_size: int = 0,
    ) -> None:
        """Initialize SwinTransformerBlockV2.

        Args:
            dim: Number of input channels.
            input_resolution: Input resolution (height, width).
            num_heads: Number of attention heads.
            window_size: Window size. Defaults to 7.
            shift_size: Shift size for SW-MSA. Defaults to 0.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
            qkv_bias: If True, add learnable bias to Q, V. Defaults to True.
            drop: Dropout probability. Defaults to 0.0.
            attn_drop: Attention dropout probability. Defaults to 0.0.
            drop_path: Stochastic depth rate. Defaults to 0.0.
            act_layer: Activation layer class. Defaults to GELU.
            norm_layer: Normalization layer class. Defaults to LayerNorm.
            pretrained_window_size: Window size in pre-training. Defaults to 0.
        """
        super(SwinTransformerBlockV2, self).__init__()
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

        # Post-norm: norm layers placed after attention and MLP
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionV2(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
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
            img_mask = torch.zeros((1, H, W, 1))
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

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Swin Transformer V2 block.

        Note: V2 uses post-norm instead of pre-norm.

        Args:
            x: Input features of shape (B, L, C) where L = H * W.

        Returns:
            Output features of shape (B, L, C).
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature has wrong size, expected {H * W}, got {L}"

        shortcut = x
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # Post-norm: apply norm after attention (different from V1)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN with post-norm
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, "
            f"num_heads={self.num_heads}, window_size={self.window_size}, "
            f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )


class PatchMergingV2(Module):
    """Patch Merging Layer for Swin Transformer V2.

    Uses post-norm order (reduction -> norm) instead of pre-norm (norm -> reduction).
    This matches the post-norm design philosophy of SwinV2.

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
        """Initialize PatchMergingV2 layer.

        Args:
            input_resolution: Input resolution (height, width).
            dim: Number of input channels.
            norm_layer: Normalization layer class. Defaults to LayerNorm.
        """
        super(PatchMergingV2, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)  # Note: V2 normalizes output (2*dim), not input (4*dim)

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

        # Post-norm order: reduction first, then norm
        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayerV2(Module):
    """A basic Swin Transformer V2 layer for one stage.

    Contains multiple SwinTransformerBlockV2 and an optional downsample layer.

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
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        norm_layer: type = LayerNorm,
        downsample: Optional[type] = None,
        use_checkpoint: bool = False,
        pretrained_window_size: int = 0,
    ) -> None:
        """Initialize BasicLayerV2.

        Args:
            dim: Number of input channels.
            input_resolution: Input resolution (height, width).
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
            qkv_bias: If True, add learnable bias to Q, V. Defaults to True.
            drop: Dropout probability. Defaults to 0.0.
            attn_drop: Attention dropout probability. Defaults to 0.0.
            drop_path: Stochastic depth rate. Defaults to 0.0.
            norm_layer: Normalization layer class. Defaults to LayerNorm.
            downsample: Downsample layer at the end. Defaults to None.
            use_checkpoint: Whether to use gradient checkpointing. Defaults to False.
            pretrained_window_size: Window size in pre-training. Defaults to 0.
        """
        super(BasicLayerV2, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlockV2(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
            )
            for i in range(depth)
        ])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through basic layer V2.

        Args:
            x: Input features of shape (B, L, C).

        Returns:
            Output features of shape (B, L', C') where L' and C' depend on downsample.
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def _init_respostnorm(self) -> None:
        """Initialize residual post-norm parameters.

        For SwinV2 post-norm, we initialize LayerNorm weights to 0,
        so that the residual branch starts as identity at the beginning of training.
        This improves training stability for deeper models.
        """
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class SwinTransformerV2(Module):
    """Swin Transformer V2 backbone for face recognition.

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
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: type = LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        pretrained_window_sizes: Optional[List[int]] = None,
        num_features: int = 512,
        feat_bn: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize SwinTransformerV2.

        Args:
            input_size: Input image size as [H, W]. Defaults to [112, 112].
            patch_size: Patch token size. Defaults to 4.
            in_chans: Number of input image channels. Defaults to 3.
            embed_dim: Patch embedding dimension. Defaults to 96.
            depths: Depth of each Swin Transformer stage. Defaults to [2, 2, 6, 2].
            num_heads: Number of attention heads in each stage. Defaults to [3, 6, 12, 24].
            window_size: Window size. Defaults to 7.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
            qkv_bias: If True, add learnable bias to Q, V. Defaults to True.
            drop_rate: Dropout rate. Defaults to 0.0.
            attn_drop_rate: Attention dropout rate. Defaults to 0.0.
            drop_path_rate: Stochastic depth rate. Defaults to 0.1.
            norm_layer: Normalization layer class. Defaults to LayerNorm.
            ape: If True, add absolute position embedding. Defaults to False.
            patch_norm: If True, add normalization after patch embedding. Defaults to True.
            use_checkpoint: Whether to use gradient checkpointing. Defaults to False.
            pretrained_window_sizes: Pretrained window sizes of each layer. Defaults to None.
            num_features: Output feature dimension. Defaults to 512.
            feat_bn: Whether to apply BatchNorm to output features. Defaults to True.
            dropout_rate: Dropout rate before final feature layer. Defaults to 0.0.
        """
        super(SwinTransformerV2, self).__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Default pretrained window sizes to 0 for each layer
        if pretrained_window_sizes is None:
            pretrained_window_sizes = [0] * self.num_layers

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
            layer = BasicLayerV2(
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
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMergingV2 if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_sizes[i_layer],
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

        self.output_layer = Sequential(
            norm_layer(self.num_features),
            Flatten(),
            Dropout(p=dropout_rate) if dropout_rate > 0 else Identity(),
            Linear(flat_dim, num_features),
            BatchNorm1d(num_features) if feat_bn else Identity(),
        )

        self.apply(self._init_weights)
        for layer in self.layers:
            layer._init_respostnorm()

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
            Set of keywords (includes V2-specific: cpb_mlp, logit_scale).
        """
        return {'cpb_mlp', 'logit_scale', 'relative_position_bias_table'}

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


def SwinV2_Tiny(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> SwinTransformerV2:
    """Create SwinV2-Tiny model for face recognition.

    - 224x224: patch_size=4, 4 stages (56 -> 28 -> 14 -> 7)
    - 112x112: patch_size=2, 4 stages (56 -> 28 -> 14 -> 7)

    Args:
        input_size: Input image size as [H, W]. Defaults to [112, 112].
        num_features: Output feature dimension. Defaults to 512.
        dropout_rate: Dropout rate. Defaults to 0.0.
        **kwargs: Additional arguments for SwinTransformerV2.

    Returns:
        SwinV2-Tiny model instance.
    """
    depths, num_heads, patch_size = _get_swin_config_by_resolution(
        input_size,
        depths_4stage=[2, 2, 6, 2],
        num_heads_4stage=[3, 6, 12, 24],
    )
    return SwinTransformerV2(
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


def SwinV2_Small(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> SwinTransformerV2:
    """Create SwinV2-Small model for face recognition.

    Args:
        input_size: Input image size as [H, W]. Defaults to [112, 112].
        num_features: Output feature dimension. Defaults to 512.
        dropout_rate: Dropout rate. Defaults to 0.0.
        **kwargs: Additional arguments for SwinTransformerV2.

    Returns:
        SwinV2-Small model instance.
    """
    depths, num_heads, patch_size = _get_swin_config_by_resolution(
        input_size,
        depths_4stage=[2, 2, 18, 2],
        num_heads_4stage=[3, 6, 12, 24],
    )
    return SwinTransformerV2(
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


def SwinV2_Base(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> SwinTransformerV2:
    """Create SwinV2-Base model for face recognition.

    Args:
        input_size: Input image size as [H, W]. Defaults to [112, 112].
        num_features: Output feature dimension. Defaults to 512.
        dropout_rate: Dropout rate. Defaults to 0.0.
        **kwargs: Additional arguments for SwinTransformerV2.

    Returns:
        SwinV2-Base model instance.
    """
    depths, num_heads, patch_size = _get_swin_config_by_resolution(
        input_size,
        depths_4stage=[2, 2, 18, 2],
        num_heads_4stage=[4, 8, 16, 32],
    )
    return SwinTransformerV2(
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


def SwinV2_Large(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> SwinTransformerV2:
    """Create SwinV2-Large model for face recognition.

    Args:
        input_size: Input image size as [H, W]. Defaults to [112, 112].
        num_features: Output feature dimension. Defaults to 512.
        dropout_rate: Dropout rate. Defaults to 0.0.
        **kwargs: Additional arguments for SwinTransformerV2.

    Returns:
        SwinV2-Large model instance.
    """
    depths, num_heads, patch_size = _get_swin_config_by_resolution(
        input_size,
        depths_4stage=[2, 2, 18, 2],
        num_heads_4stage=[6, 12, 24, 48],
    )
    return SwinTransformerV2(
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
