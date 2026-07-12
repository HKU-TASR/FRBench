"""ConvNeXt and ConvNeXtV2 backbones for face recognition.

Reference Papers:
    - ConvNeXt:   A ConvNet for the 2020s (https://arxiv.org/abs/2201.03545)
    - ConvNeXtV2: Co-designing and Scaling ConvNets with Masked Autoencoders
                  (https://arxiv.org/abs/2301.00808)
Reference Code: facebookresearch/ConvNeXt, facebookresearch/ConvNeXt-V2, timm.

Supported Variants:
    ConvNeXt:   T, S, B, L, XL
    ConvNeXtV2: Atto, Femto, Pico, Nano, T, S, B, L, H
"""
from typing import List

import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm1d,
    Linear,
    GELU,
    Identity,
)

from .utils import DropPath, LayerNorm2d, build_fr_head


def _get_convnext_stem_kwargs(input_size: List[int]) -> dict:
    """Resolution-aware stem patchify stride for ConvNeXt-family backbones.

    Args:
        input_size: ``[H, W]`` of the input image. H == W is assumed.

    Returns:
        ``{"stem_patch_size": 2}`` for face-recognition inputs (H <= 112),
        ``{"stem_patch_size": 4}`` for ImageNet-class inputs (>= 224).
    """
    assert input_size[0] == input_size[1], (
        f"ConvNeXt expects a square input, got {input_size}."
    )
    if input_size[0] <= 112:
        return {"stem_patch_size": 2}
    return {"stem_patch_size": 4}


__all__ = [
    "ConvNeXt",
    "ConvNeXtV2",
    # ConvNeXt Variants
    "ConvNeXt_Tiny",
    "ConvNeXt_Small",
    "ConvNeXt_Base",
    "ConvNeXt_Large",
    "ConvNeXt_XLarge",
    # ConvNeXtV2 Variants
    "ConvNeXtV2_Atto",
    "ConvNeXtV2_Femto",
    "ConvNeXtV2_Pico",
    "ConvNeXtV2_Nano",
    "ConvNeXtV2_Tiny",
    "ConvNeXtV2_Small",
    "ConvNeXtV2_Base",
    "ConvNeXtV2_Large",
    "ConvNeXtV2_Huge",
]


class GlobalResponseNorm(Module):
    """Global Response Normalization layer from ConvNeXt-V2.
    
    Attributes:
        gamma: Learnable scale parameter.
        beta: Learnable shift parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize GlobalResponseNorm.
        
        Args:
            dim: Number of channels.
            eps: Small constant for numerical stability. Defaults to 1e-6.
        """
        super(GlobalResponseNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global response normalization.
        
        Args:
            x: Input tensor of shape (N, H, W, C) in channels-last format.
            
        Returns:
            torch.Tensor: Normalized tensor of shape (N, H, W, C).
        """
        # Compute L2 norm over spatial dimensions (H, W) in NHWC layout
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        # Normalize by mean of norms across channels
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return x + torch.addcmul(self.beta, self.gamma, x * nx)


class ConvNeXtBlock(Module):
    """ConvNeXt block with depthwise conv, layer norm, and inverted bottleneck MLP.
    
    Architecture:
        DwConv (7x7) -> LayerNorm -> Linear (expand) -> GELU -> Linear (project)
        
    For ConvNeXtV2, includes Global Response Normalization after GELU.
    
    Attributes:
        dwconv: Depthwise convolution layer.
        norm: Layer normalization.
        pwconv1: First pointwise convolution (channel expansion).
        act: Activation function (GELU).
        grn: Global response normalization (only for V2).
        pwconv2: Second pointwise convolution (channel projection).
        gamma: Layer scale parameter.
        drop_path: Stochastic depth module.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        use_grn: bool = False,
    ) -> None:
        """Initialize ConvNeXtBlock.
        
        Args:
            dim: Number of input/output channels.
            drop_path: Drop path rate. Defaults to 0.0.
            layer_scale_init_value: Initial value for layer scale. 
                Defaults to 1e-6. Set to 0 or None to disable.
            mlp_ratio: MLP hidden dimension expansion ratio. Defaults to 4.0.
            use_grn: Whether to use Global Response Normalization (V2). 
                Defaults to False.
        """
        super(ConvNeXtBlock, self).__init__()
        hidden_dim = int(dim * mlp_ratio)
        
        # Depthwise convolution
        self.dwconv = Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # Layer normalization (channels-last)
        self.norm = nn.LayerNorm(dim)
        # Pointwise linear layers (inverted bottleneck, operate in NHWC)
        self.pwconv1 = Linear(dim, hidden_dim)
        self.act = GELU()
        # Global Response Normalization (ConvNeXtV2 only, operates in NHWC)
        self.grn = GlobalResponseNorm(hidden_dim) if use_grn else Identity()
        self.pwconv2 = Linear(hidden_dim, dim)
        
        # Layer scale
        if layer_scale_init_value is not None and layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True
            )
        else:
            self.gamma = None
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ConvNeXt block.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W).
        """
        shortcut = x
        # Depthwise conv (NCHW)
        x = self.dwconv(x)
        # Permute to channels-last (NHWC) for norm + linear layers
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        # Inverted bottleneck MLP (operates in NHWC)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        # Layer scale
        if self.gamma is not None:
            x = x * self.gamma
        # Permute back to channels-first (NCHW)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        # Residual connection with drop path
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtStage(Module):
    """ConvNeXt stage consisting of downsampling and multiple blocks.
    
    Attributes:
        downsample: Downsampling layer (norm + conv).
        blocks: Sequential container of ConvNeXt blocks.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        drop_path_rates: List[float],
        layer_scale_init_value: float = 1e-6,
        downsample: bool = True,
        use_grn: bool = False,
    ) -> None:
        """Initialize ConvNeXtStage.
        
        Args:
            in_dim: Number of input channels.
            out_dim: Number of output channels.
            depth: Number of blocks in this stage.
            drop_path_rates: Drop path rates for each block.
            layer_scale_init_value: Initial value for layer scale. 
                Defaults to 1e-6.
            downsample: Whether to downsample at stage start. Defaults to True.
            use_grn: Whether to use Global Response Normalization (V2). 
                Defaults to False.
        """
        super(ConvNeXtStage, self).__init__()
        
        # Downsampling layer
        if downsample:
            self.downsample = Sequential(
                LayerNorm2d(in_dim),
                Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
            )
        else:
            self.downsample = Identity()
        
        # Build blocks
        blocks = []
        for i in range(depth):
            blocks.append(
                ConvNeXtBlock(
                    dim=out_dim,
                    drop_path=drop_path_rates[i],
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                )
            )
        self.blocks = Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the stage.
        
        Args:
            x: Input tensor of shape (N, C_in, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H/2, W/2).
        """
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class ConvNeXt(Module):
    """ConvNeXt backbone for face recognition.

    Architecture:
        - Patchify stem (k=stem_patch_size, s=stem_patch_size) + LayerNorm2d.
            * stem_patch_size=4 for 224 input (official),
            * stem_patch_size=2 for 112 input (face-recognition crops).
        - 4 stages with increasing channels and decreasing resolution. Each
          stage = LayerNorm2d + Conv2d(k=2, s=2) downsample (stages 1..3) +
          N ConvNeXtBlocks. Same as official.
        - Position-preserving FR head:
              LayerNorm2d(C) -> Flatten -> Dropout? -> Linear(7*7*C, 512) -> BN1d.
          Same pattern as `swin_v1.py`, `swin_v2.py`, `swin_mlp.py`.

    Attributes:
        stem: Patchify stem (Conv2d + LayerNorm2d).
        stages: ``nn.ModuleList`` of four ConvNeXt stages.
        output_layer: FaceX-Zoo-style position-preserving embedding head.
        final_resolution: ``(H_final, W_final)`` of the last-stage feature map.
    """

    def __init__(
        self,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        input_size: List[int] = [112, 112],
        stem_patch_size: int = 4,
        num_features: int = 512,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
        use_grn: bool = False,
        feat_bn: bool = True,
    ) -> None:
        """Initialize ConvNeXt backbone.

        Keyword Args:
            depths (List[int]): Number of blocks in each of the 4 stages.
                Defaults to [3, 3, 9, 3] (ConvNeXt-Tiny).
            dims (List[int]): Feature dimensions for each stage.
                Defaults to [96, 192, 384, 768] (ConvNeXt-Tiny).
            input_size (List[int]): Input image size as [H, W]. Defaults to
                [112, 112]. Combined with ``stem_patch_size`` it determines the
                final spatial grid that the position-preserving head flattens.
            stem_patch_size (int): Kernel / stride of the patchify stem.
                4 for 224-class inputs (official), 2 for 112-class inputs.
                Factory helpers call ``_get_convnext_stem_kwargs`` to set this
                automatically based on ``input_size``.
            num_features (int): Output embedding dimension. Defaults to 512.
            dropout_rate (float): Dropout probability inserted in the head
                between the flatten and the Linear projection. Defaults to 0.0
                (BN-Neck already regularizes; matches the Swin family).
            drop_path_rate (float): Maximum stochastic depth rate (linear ramp
                across blocks). Defaults to 0.1. Override per-variant in the
                YAML config (FR recipes lower this vs. the ImageNet 0.5).
            layer_scale_init_value (float): Initial value for LayerScale
                ``gamma``. Defaults to 1e-6. Set to ``None`` or 0 to disable.
                ConvNeXtV2 disables it; GRN takes over its regularizing role.
            use_grn (bool): Whether to insert Global Response Normalization
                between GELU and pwconv2 in every block (ConvNeXtV2).
                Defaults to False.
            feat_bn (bool): Whether to append a BatchNorm1d (BN-Neck) on the
                512-d embedding. Defaults to True (FaceX-Zoo convention).
        """
        super(ConvNeXt, self).__init__()

        assert len(depths) == 4 and len(dims) == 4, (
            f"ConvNeXt expects 4 stages, got depths={depths}, dims={dims}."
        )
        assert input_size[0] == input_size[1], (
            f"ConvNeXt expects a square input, got {input_size}."
        )

        self.num_features = num_features
        self.depths = depths
        self.dims = dims

        # Patchify stem -- non-overlapping Conv2d(k=stem_patch_size, s=stem_patch_size).
        # 112 with sps=2 -> 56x56; 224 with sps=4 -> 56x56 (official ImageNet behavior).
        self.stem = Sequential(
            Conv2d(3, dims[0], kernel_size=stem_patch_size, stride=stem_patch_size),
            LayerNorm2d(dims[0], eps=1e-6),
        )

        # Per-block drop-path rates (linear ramp 0 -> drop_path_rate across all blocks).
        total_depth = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # 4 stages. Stage 0 has no internal downsample (the stem already produced
        # 56x56); stages 1..3 each apply LayerNorm2d + Conv2d(k=2, s=2).
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = ConvNeXtStage(
                in_dim=dims[i - 1] if i > 0 else dims[0],
                out_dim=dims[i],
                depth=depths[i],
                drop_path_rates=dp_rates[cur:cur + depths[i]],
                layer_scale_init_value=layer_scale_init_value,
                downsample=(i > 0),
                use_grn=use_grn,
            )
            self.stages.append(stage)
            cur += depths[i]

        final_h = input_size[0] // (stem_patch_size * 8)
        final_w = input_size[1] // (stem_patch_size * 8)
        assert final_h > 0 and final_w > 0, (
            f"Final-stage resolution must be positive, got ({final_h}, {final_w}). "
            f"Check that input_size / stem_patch_size produces enough downsampling stages."
        )
        self.final_resolution = (final_h, final_w)

        self.output_layer = build_fr_head(
            in_channels=dims[-1],
            spatial_size=self.final_resolution,
            num_features=num_features,
            dropout_rate=dropout_rate,
            feat_bn=feat_bn,
        )

        # Init: trunc_normal_(std=0.02) on Conv/Linear weights, zero bias; LN/BN to 1/0.
        self.apply(self._init_weights)

    def _init_weights(self, m: Module) -> None:
        """Initialize per-module weights (called via ``self.apply``).
        """
        if isinstance(m, (Conv2d, Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (BatchNorm1d, LayerNorm2d, nn.LayerNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> set:
        return {"grn.gamma", "grn.beta"}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the final-stage feature map.

        Args:
            x: Input image of shape ``(B, 3, H, W)``.

        Returns:
            Feature map of shape ``(B, dims[-1], H_final, W_final)``.
        """
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embeddings from input images.

        Args:
            x: Input images of shape ``(B, 3, H, W)``.

        Returns:
            Face embeddings of shape ``(B, num_features)``.
        """
        x = self.forward_features(x)
        x = self.output_layer(x)
        return x


class ConvNeXtV2(ConvNeXt):
    """ConvNeXtV2 backbone for face recognition.
    """

    def __init__(
        self,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        input_size: List[int] = [112, 112],
        stem_patch_size: int = 4,
        num_features: int = 512,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        feat_bn: bool = True,
    ) -> None:
        """Initialize ConvNeXtV2 backbone. See :class:`ConvNeXt` for arg docs.

        V2-defining arguments removed from the signature:
        ``layer_scale_init_value`` is forced to ``None`` and ``use_grn`` is
        forced to ``True``. These are the architectural deltas that define V2
        in the paper and should not be configurable here.
        """
        super(ConvNeXtV2, self).__init__(
            depths=depths,
            dims=dims,
            input_size=input_size,
            stem_patch_size=stem_patch_size,
            num_features=num_features,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            feat_bn=feat_bn,
            layer_scale_init_value=None,  # V2 removes LayerScale (paper Sec. 3.3)
            use_grn=True,                  # V2 adds GRN between GELU and pwconv2
        )


# ============================================================================
# Factory Functions
# ----------------------------------------------------------------------------
# All factories share a uniform Swin-family-style signature:
#   (input_size=[112,112], num_features=512, dropout_rate=0.0, **kwargs).
# ============================================================================


# ConvNeXt Variants -----------------------------------------------------------

def ConvNeXt_Tiny(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> ConvNeXt:
    """Construct a ConvNeXt-Tiny model (~28M backbone params).

    Architecture: depths=[3, 3, 9, 3], dims=[96, 192, 384, 768].
    """
    return ConvNeXt(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        input_size=input_size,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **_get_convnext_stem_kwargs(input_size),
        **kwargs,
    )


def ConvNeXt_Small(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> ConvNeXt:
    """Construct a ConvNeXt-Small model (~50M backbone params).

    Architecture: depths=[3, 3, 27, 3], dims=[96, 192, 384, 768].
    """
    return ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        input_size=input_size,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **_get_convnext_stem_kwargs(input_size),
        **kwargs,
    )


def ConvNeXt_Base(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> ConvNeXt:
    """Construct a ConvNeXt-Base model (~89M backbone params).

    Architecture: depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024].
    """
    return ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        input_size=input_size,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **_get_convnext_stem_kwargs(input_size),
        **kwargs,
    )


def ConvNeXt_Large(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> ConvNeXt:
    """Construct a ConvNeXt-Large model (~198M backbone params).

    Architecture: depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536].
    """
    return ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        input_size=input_size,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **_get_convnext_stem_kwargs(input_size),
        **kwargs,
    )


def ConvNeXt_XLarge(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> ConvNeXt:
    """Construct a ConvNeXt-XLarge model (~350M backbone params).

    Architecture: depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048].
    """
    return ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[256, 512, 1024, 2048],
        input_size=input_size,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **_get_convnext_stem_kwargs(input_size),
        **kwargs,
    )


# ConvNeXtV2 Variants ---------------------------------------------------------

def ConvNeXtV2_Atto(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> ConvNeXtV2:
    """Construct a ConvNeXtV2-Atto model (~3.7M backbone params).

    Architecture: depths=[2, 2, 6, 2], dims=[40, 80, 160, 320].
    """
    return ConvNeXtV2(
        depths=[2, 2, 6, 2],
        dims=[40, 80, 160, 320],
        input_size=input_size,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **_get_convnext_stem_kwargs(input_size),
        **kwargs,
    )


def ConvNeXtV2_Femto(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> ConvNeXtV2:
    """Construct a ConvNeXtV2-Femto model (~5.2M backbone params).

    Architecture: depths=[2, 2, 6, 2], dims=[48, 96, 192, 384].
    """
    return ConvNeXtV2(
        depths=[2, 2, 6, 2],
        dims=[48, 96, 192, 384],
        input_size=input_size,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **_get_convnext_stem_kwargs(input_size),
        **kwargs,
    )


def ConvNeXtV2_Pico(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> ConvNeXtV2:
    """Construct a ConvNeXtV2-Pico model (~9.1M backbone params).

    Architecture: depths=[2, 2, 6, 2], dims=[64, 128, 256, 512].
    """
    return ConvNeXtV2(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 256, 512],
        input_size=input_size,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **_get_convnext_stem_kwargs(input_size),
        **kwargs,
    )


def ConvNeXtV2_Nano(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> ConvNeXtV2:
    """Construct a ConvNeXtV2-Nano model (~15.6M backbone params).

    Architecture: depths=[2, 2, 8, 2], dims=[80, 160, 320, 640].
    Note the [2, 2, 8, 2] stage-3 depth (vs Atto/Femto/Pico's 6).
    """
    return ConvNeXtV2(
        depths=[2, 2, 8, 2],
        dims=[80, 160, 320, 640],
        input_size=input_size,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **_get_convnext_stem_kwargs(input_size),
        **kwargs,
    )


def ConvNeXtV2_Tiny(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> ConvNeXtV2:
    """Construct a ConvNeXtV2-Tiny model (~28M backbone params).

    Architecture: depths=[3, 3, 9, 3], dims=[96, 192, 384, 768].
    """
    return ConvNeXtV2(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        input_size=input_size,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **_get_convnext_stem_kwargs(input_size),
        **kwargs,
    )


def ConvNeXtV2_Small(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> ConvNeXtV2:
    """Construct a ConvNeXtV2-Small model (~50M backbone params).

    Architecture: depths=[3, 3, 27, 3], dims=[96, 192, 384, 768].
    """
    return ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        input_size=input_size,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **_get_convnext_stem_kwargs(input_size),
        **kwargs,
    )


def ConvNeXtV2_Base(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> ConvNeXtV2:
    """Construct a ConvNeXtV2-Base model (~89M backbone params).

    Architecture: depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024].
    """
    return ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        input_size=input_size,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **_get_convnext_stem_kwargs(input_size),
        **kwargs,
    )


def ConvNeXtV2_Large(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> ConvNeXtV2:
    """Construct a ConvNeXtV2-Large model (~198M backbone params).

    Architecture: depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536].
    """
    return ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        input_size=input_size,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **_get_convnext_stem_kwargs(input_size),
        **kwargs,
    )


def ConvNeXtV2_Huge(
    input_size: List[int] = [112, 112],
    num_features: int = 512,
    dropout_rate: float = 0.0,
    **kwargs,
) -> ConvNeXtV2:
    """Construct a ConvNeXtV2-Huge model (~660M backbone params).

    Architecture: depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816].
    """
    return ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[352, 704, 1408, 2816],
        input_size=input_size,
        num_features=num_features,
        dropout_rate=dropout_rate,
        **_get_convnext_stem_kwargs(input_size),
        **kwargs,
    )