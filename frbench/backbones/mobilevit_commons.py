"""Shared building blocks for the MobileViT backbone family (V1 / V2 / V3).

Reference Papers:
    - MobileViT (V1): MobileViT: Light-weight, General-purpose, and Mobile-friendly
                      Vision Transformer (https://arxiv.org/abs/2110.02178)
    - MobileViTv2:    Separable Self-attention for Mobile Vision Transformers
                      (https://arxiv.org/abs/2206.02680)
    - MobileViTv3:    MobileViTv3: Mobile-Friendly Vision Transformer with Simple and
                      Effective Fusion of Local, Global and Input Features
                      (https://arxiv.org/abs/2209.15159)
Reference Code:
    - https://github.com/apple/ml-cvnets
    - https://github.com/micronDLA/MobileViTv3
    - https://github.com/jaiwei98/mobile-vit-pytorch

Imported by ``mobilevit_v1.py``, ``mobilevit_v2.py`` and ``mobilevit_v3.py``.
"""
import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm2d,
    GroupNorm,
    LayerNorm,
    Linear,
    SiLU,
    Softmax,
    Dropout,
    Identity,
)


__all__ = [
    "ConvLayer",
    "InvertedResidual",
    "MultiHeadAttention",
    "TransformerEncoder",
    "LinearSelfAttention",
    "LinearAttnFFN",
    "unfolding_interpolate",
    "folding_interpolate",
    "unfolding_unfold",
    "folding_fold",
]


class ConvLayer(Module):
    """Conv2d -> (BatchNorm2d) -> (SiLU/Swish), the basic MobileViT conv unit.

    Padding is auto-set to ``(kernel_size - 1) // 2`` so odd kernels keep the
    spatial size ('same' convolution), matching the official cvnets ``ConvLayer``.

    Attributes:
        conv: Convolution layer (no bias when followed by BatchNorm).
        norm: BatchNorm2d or Identity.
        act: SiLU (Swish) or Identity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        use_norm: bool = True,
        use_act: bool = True,
    ) -> None:
        """Initialize ConvLayer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size. Defaults to 3.
            stride: Convolution stride. Defaults to 1.
            groups: Number of groups (set to in_channels for depthwise). Defaults to 1.
            bias: Whether the conv has a bias term. Defaults to False.
            use_norm: Whether to append BatchNorm2d. Defaults to True.
            use_act: Whether to append the SiLU activation. Defaults to True.
        """
        super(ConvLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=bias,
        )
        self.norm = BatchNorm2d(out_channels) if use_norm else Identity()
        self.act = SiLU() if use_act else Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv -> norm -> activation.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Output tensor of shape (B, out_channels, H', W').
        """
        return self.act(self.norm(self.conv(x)))


class InvertedResidual(Module):
    """MobileNetV2 inverted residual block (the MV2 block used for downsampling).

    Pointwise expand (1x1) -> depthwise (3x3, stride) -> pointwise project (1x1,
    linear). A residual connection is used only when stride == 1 and the channel
    count is preserved. Matches the official MobileViT MV2 block.

    Attributes:
        use_res_connect: Whether the identity skip connection is active.
        block: The inverted-residual conv stack.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        """Initialize InvertedResidual.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Depthwise stride (1 or 2).
            expand_ratio: Channel expansion factor for the hidden dimension.
        """
        super(InvertedResidual, self).__init__()
        assert stride in (1, 2), f"InvertedResidual stride must be 1 or 2, got {stride}."
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        layers: List[Module] = []
        if expand_ratio != 1:
            layers.append(ConvLayer(in_channels, hidden_dim, kernel_size=1))
        layers.append(
            ConvLayer(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim)
        )
        layers.append(ConvLayer(hidden_dim, out_channels, kernel_size=1, use_act=False))
        self.block = Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Output tensor of shape (B, out_channels, H', W').
        """
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class MultiHeadAttention(Module):
    """Standard scaled dot-product multi-head self-attention over patch tokens.

    Used by the MobileViT V1 / V3-V1 global representation. Operates on token
    sequences of shape (B, N, C) where C is split into ``num_heads`` heads.

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Per-head channel dimension (C // num_heads).
        scale: Query scaling factor (head_dim ** -0.5).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Initialize MultiHeadAttention.

        Args:
            embed_dim: Token embedding dimension.
            num_heads: Number of attention heads. Defaults to 4.
            attn_dropout: Dropout on the attention matrix. Defaults to 0.0.
            bias: Whether QKV / output projections use bias. Defaults to True.
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
        )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.attn_dropout = Dropout(attn_dropout)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.softmax = Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            x: Token tensor of shape (B, N, C).

        Returns:
            Output tokens of shape (B, N, C).
        """
        b, n, c = x.shape
        qkv = self.qkv_proj(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = self.softmax((q * self.scale) @ k.transpose(-2, -1))
        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.out_proj(out)


class TransformerEncoder(Module):
    """Pre-norm Transformer encoder (MHSA + FFN) for MobileViT V1 / V3-V1.

    LayerNorm -> MHSA -> residual; LayerNorm -> FFN (Linear-SiLU-Linear) ->
    residual. Matches the official MobileViT transformer block.

    Attributes:
        pre_norm_mha: Pre-norm multi-head-attention branch.
        pre_norm_ffn: Pre-norm feed-forward branch.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        """Initialize TransformerEncoder.

        Args:
            embed_dim: Token embedding dimension.
            ffn_latent_dim: Hidden dimension of the FFN.
            num_heads: Number of attention heads. Defaults to 4.
            dropout: Dropout after attention and inside the FFN. Defaults to 0.0.
            attn_dropout: Dropout on the attention matrix. Defaults to 0.0.
        """
        super(TransformerEncoder, self).__init__()
        self.pre_norm_mha = Sequential(
            LayerNorm(embed_dim, eps=1e-5),
            MultiHeadAttention(embed_dim, num_heads, attn_dropout),
            Dropout(dropout),
        )
        self.pre_norm_ffn = Sequential(
            LayerNorm(embed_dim, eps=1e-5),
            Linear(embed_dim, ffn_latent_dim),
            SiLU(),
            Dropout(dropout),
            Linear(ffn_latent_dim, embed_dim),
            Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Token tensor of shape (B, N, C).

        Returns:
            Output tokens of shape (B, N, C).
        """
        x = x + self.pre_norm_mha(x)
        x = x + self.pre_norm_ffn(x)
        return x


class LinearSelfAttention(Module):
    """Separable self-attention with linear complexity (MobileViTv2).

    Replaces the quadratic MHSA with an O(k) context-vector aggregation. Operates
    directly on unfolded feature maps of shape (B, C, P, N): a single-channel
    query produces context scores over the N patches, which weight the keys into
    one context vector that modulates the (ReLU) values. Matches the official
    cvnets implementation (Eq. in arXiv:2206.02680).

    Attributes:
        qkv_proj: 1x1 conv producing [query(1), key(C), value(C)].
        out_proj: 1x1 output projection.
        embed_dim: Channel dimension C.
    """

    def __init__(self, embed_dim: int, attn_dropout: float = 0.0, bias: bool = True) -> None:
        """Initialize LinearSelfAttention.

        Args:
            embed_dim: Channel dimension of the unfolded patches.
            attn_dropout: Dropout on the context scores. Defaults to 0.0.
            bias: Whether the 1x1 projections use bias. Defaults to True.
        """
        super(LinearSelfAttention, self).__init__()
        self.qkv_proj = Conv2d(embed_dim, 1 + 2 * embed_dim, kernel_size=1, bias=bias)
        self.attn_dropout = Dropout(attn_dropout)
        self.out_proj = Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply separable self-attention.

        Args:
            x: Unfolded patches of shape (B, C, P, N).

        Returns:
            Output of shape (B, C, P, N).
        """
        qkv = self.qkv_proj(x)
        query, key, value = torch.split(qkv, [1, self.embed_dim, self.embed_dim], dim=1)
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)
        context_vector = torch.sum(key * context_scores, dim=-1, keepdim=True)
        out = F.relu(value) * context_vector.expand_as(value)
        return self.out_proj(out)


class LinearAttnFFN(Module):
    """Pre-norm separable-attention + FFN block (MobileViTv2 / V3-V2).

    Mirrors ``TransformerEncoder`` but works on (B, C, P, N) feature maps: the
    normalization is GroupNorm with a single group (== LayerNorm over channels),
    attention is the linear ``LinearSelfAttention``, and the FFN is built from
    1x1 convolutions. Matches the official cvnets ``LinearAttnFFN``.

    Attributes:
        pre_norm_attn: Pre-norm separable-attention branch.
        pre_norm_ffn: Pre-norm convolutional feed-forward branch.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        """Initialize LinearAttnFFN.

        Args:
            embed_dim: Channel dimension of the unfolded patches.
            ffn_latent_dim: Hidden dimension of the FFN.
            dropout: Dropout after attention and inside the FFN. Defaults to 0.0.
            attn_dropout: Dropout on the context scores. Defaults to 0.0.
        """
        super(LinearAttnFFN, self).__init__()
        self.pre_norm_attn = Sequential(
            GroupNorm(num_groups=1, num_channels=embed_dim, eps=1e-5),
            LinearSelfAttention(embed_dim, attn_dropout),
            Dropout(dropout),
        )
        self.pre_norm_ffn = Sequential(
            GroupNorm(num_groups=1, num_channels=embed_dim, eps=1e-5),
            Conv2d(embed_dim, ffn_latent_dim, kernel_size=1),
            SiLU(),
            Dropout(dropout),
            Conv2d(ffn_latent_dim, embed_dim, kernel_size=1),
            Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Unfolded patches of shape (B, C, P, N).

        Returns:
            Output of shape (B, C, P, N).
        """
        x = x + self.pre_norm_attn(x)
        x = x + self.pre_norm_ffn(x)
        return x


def unfolding_interpolate(
    feature_map: torch.Tensor, patch_h: int, patch_w: int
) -> Tuple[torch.Tensor, Dict]:
    """Unfold a feature map into patch tokens (MobileViT V1 / V3-V1 style).

    The map is bilinearly resized up to a multiple of the patch size when needed
    (the official cvnets behaviour for non-divisible resolutions, e.g. the 7x7
    grid produced by a 112x112 face crop) and reshaped to ``(B * P, N, C)`` so a
    standard Transformer attends across the N patches for each of the P
    in-patch pixel positions.

    Args:
        feature_map: Tensor of shape (B, C, H, W).
        patch_h: Patch height.
        patch_w: Patch width.

    Returns:
        A tuple ``(patches, info_dict)`` where ``patches`` has shape (B*P, N, C)
        and ``info_dict`` carries the metadata needed by ``folding_interpolate``.
    """
    patch_area = patch_h * patch_w
    b, c, orig_h, orig_w = feature_map.shape

    new_h = int(math.ceil(orig_h / patch_h) * patch_h)
    new_w = int(math.ceil(orig_w / patch_w) * patch_w)
    interpolate = new_h != orig_h or new_w != orig_w
    if interpolate:
        feature_map = F.interpolate(
            feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False
        )

    num_patch_h = new_h // patch_h
    num_patch_w = new_w // patch_w
    num_patches = num_patch_h * num_patch_w

    # (B, C, H, W) -> (B*C*n_h, p_h, n_w, p_w) -> (B*C*n_h, n_w, p_h, p_w)
    x = feature_map.reshape(b * c * num_patch_h, patch_h, num_patch_w, patch_w)
    x = x.transpose(1, 2)
    # -> (B, C, N, P) -> (B, P, N, C) -> (B*P, N, C)
    x = x.reshape(b, c, num_patches, patch_area)
    x = x.transpose(1, 3)
    patches = x.reshape(b * patch_area, num_patches, -1)

    info_dict = {
        "orig_size": (orig_h, orig_w),
        "batch_size": b,
        "interpolate": interpolate,
        "total_patches": num_patches,
        "num_patches_h": num_patch_h,
        "num_patches_w": num_patch_w,
    }
    return patches, info_dict


def folding_interpolate(
    patches: torch.Tensor, info_dict: Dict, patch_h: int, patch_w: int
) -> torch.Tensor:
    """Fold patch tokens back into a feature map (inverse of ``unfolding_interpolate``).

    Args:
        patches: Tensor of shape (B*P, N, C).
        info_dict: Metadata produced by ``unfolding_interpolate``.
        patch_h: Patch height.
        patch_w: Patch width.

    Returns:
        Feature map of shape (B, C, H, W) restored to the original resolution.
    """
    patch_area = patch_h * patch_w
    b = info_dict["batch_size"]
    num_patches = info_dict["total_patches"]
    num_patch_h = info_dict["num_patches_h"]
    num_patch_w = info_dict["num_patches_w"]

    x = patches.contiguous().view(b, patch_area, num_patches, -1)
    x = x.transpose(1, 3)
    channels = x.shape[1]
    x = x.reshape(b * channels * num_patch_h, num_patch_w, patch_h, patch_w)
    x = x.transpose(1, 2)
    feature_map = x.reshape(b, channels, num_patch_h * patch_h, num_patch_w * patch_w)

    if info_dict["interpolate"]:
        feature_map = F.interpolate(
            feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False
        )
    return feature_map


def unfolding_unfold(
    feature_map: torch.Tensor, patch_h: int, patch_w: int
) -> Tuple[torch.Tensor, Dict]:
    """Unfold a feature map into patches via ``F.unfold`` (MobileViT V2 / V3-V2 style).

    Produces patches of shape (B, C, P, N) consumed directly by the separable
    attention. As in ``unfolding_interpolate`` the map is resized to a multiple
    of the patch size when needed so the 7x7 grid from a 112x112 face crop folds
    back losslessly to its original resolution.

    Args:
        feature_map: Tensor of shape (B, C, H, W).
        patch_h: Patch height.
        patch_w: Patch width.

    Returns:
        A tuple ``(patches, info_dict)`` where ``patches`` has shape (B, C, P, N)
        and ``info_dict`` carries the sizes needed by ``folding_fold``.
    """
    b, c, orig_h, orig_w = feature_map.shape

    new_h = int(math.ceil(orig_h / patch_h) * patch_h)
    new_w = int(math.ceil(orig_w / patch_w) * patch_w)
    interpolate = new_h != orig_h or new_w != orig_w
    if interpolate:
        feature_map = F.interpolate(
            feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False
        )

    patches = F.unfold(feature_map, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
    patches = patches.reshape(b, c, patch_h * patch_w, -1)

    info_dict = {
        "orig_size": (orig_h, orig_w),
        "resized_size": (new_h, new_w),
        "interpolate": interpolate,
    }
    return patches, info_dict


def folding_fold(
    patches: torch.Tensor, info_dict: Dict, patch_h: int, patch_w: int
) -> torch.Tensor:
    """Fold patches back into a feature map (inverse of ``unfolding_unfold``).

    Args:
        patches: Tensor of shape (B, C, P, N).
        info_dict: Metadata produced by ``unfolding_unfold``.
        patch_h: Patch height.
        patch_w: Patch width.

    Returns:
        Feature map of shape (B, C, H, W) restored to the original resolution.
    """
    b, c, p, n = patches.shape
    # (B, C, P, N) -> (B, C * P, N) -> fold -> (B, C, H', W')
    patches = patches.reshape(b, c * p, n)
    feature_map = F.fold(
        patches,
        output_size=info_dict["resized_size"],
        kernel_size=(patch_h, patch_w),
        stride=(patch_h, patch_w),
    )
    if info_dict["interpolate"]:
        feature_map = F.interpolate(
            feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False
        )
    return feature_map
