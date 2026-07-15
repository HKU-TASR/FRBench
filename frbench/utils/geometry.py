"""Differentiable face-alignment and cropping geometry.

These are standalone, publicly exported utilities (also available as
``frbench.align``, ``frbench.crop``, etc.). They operate on plain tensors and
preserve the input value range, so they compose with any downstream model.
"""
from __future__ import annotations

from typing import Tuple
import math

import torch
import torch.nn.functional as F

from .log import warn

#: Canonical InsightFace/ArcFace 5-point template for a 112x112 crop
#: (left eye, right eye, nose, left mouth corner, right mouth corner), in
#: output-pixel coordinates.
ARCFACE_112_TEMPLATE = torch.tensor(
    [
        [38.29459953, 51.69630051],
        [73.53179932, 51.50139999],
        [56.02519989, 71.73660278],
        [41.54930115, 92.36550140],
        [70.72990036, 92.20410156],
    ],
    dtype=torch.float32)


def arcface_template(size: Tuple[int, int] = (112, 112)) -> torch.Tensor:
    """Return the ArcFace 5-point template scaled to a target crop size.

    Args:
        size: Target crop size ``(H, W)``. The canonical 112x112 template is
            scaled linearly per axis.

    Returns:
        ``(5, 2)`` float32 template in output-pixel ``(x, y)`` coordinates.
    """
    h, w = int(size[0]), int(size[1])
    template = ARCFACE_112_TEMPLATE.clone()
    template[:, 0] *= w / 112.0
    template[:, 1] *= h / 112.0
    return template


def validate_input_range(imgs: torch.Tensor) -> None:
    """Raise on out-of-range values; warn on common [0,1]/[-1,1] mistakes.

    FRBench expects RGB float tensors in ``[0, 255]``.
    """
    amax = imgs.amax().item()
    amin = imgs.amin().item()
    if amax > 256 or amin < -1:
        raise ValueError(f"Expected values in [0,255], got [{amin:.3f}, {amax:.3f}]")
    if amax <= 2.0 and amin >= -1.5:
        warn(
            f"inputs look like [0,1] or [-1,1] normalized tensors "
            f"(range [{amin:.3f}, {amax:.3f}]); expected [0,255]."
        )


def square_boxes(boxes: torch.Tensor, loosen: float = 1.0) -> torch.Tensor:
    """Turn ``(N, 4)`` boxes into centered squares, optionally enlarged.

    Args:
        boxes: ``(N, 4)`` boxes ``[x1, y1, x2, y2]`` in pixels.
        loosen: Enlargement factor for the square side length.

    Returns:
        ``(N, 4)`` square boxes centered on the originals.
    """
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = torch.maximum(x2 - x1, y2 - y1) * loosen / 2
    return torch.stack([cx - half, cy - half, cx + half, cy + half], dim=-1)


def crop(
    img: torch.Tensor,
    boxes: torch.Tensor,
    output_size: Tuple[int, int] = (112, 112),
    max_supersample: int = 4,
) -> torch.Tensor:
    """Differentiably crop and resize boxes from an image.

    Out-of-bounds regions are zero-filled. Sampling uses ``grid_sample`` so
    gradients flow back to ``img``.

    Args:
        img: ``(C, H, W)`` image.
        boxes: ``(N, 4)`` boxes ``[x1, y1, x2, y2]`` in pixels.
        output_size: Output ``(H, W)``.
        max_supersample: Maximum anti-aliasing supersample factor.

            Anti-aliasing is achieved by sampling the source at an
            ``S``-times finer internal grid (``S²`` sub-pixel samples per output
            pixel) and then box-averaging them with ``avg_pool2d(kernel=S)``.
            Plain bilinear ``grid_sample`` at the output resolution reads only a
            2×2 source neighborhood per output pixel and skips the D-2 source
            pixels in between (where D = box-width / output-size is the
            downsample ratio), which aliases. Setting S = ceil(D) spaces the
            sub-samples ≤ 1 source pixel apart so every source pixel
            contributes, and the subsequent box average becomes an exact area
            filter of width D — the correct pre-filter for downsampling.
            S = 1 is used for upsampling (D ≤ 1), which does not alias.

            This parameter caps S to bound memory: the intermediate tensor is
            ``(N, C, H_out×S, W_out×S)``, so S=8 costs 64× the memory of the
            final crop. The default of 4 handles all typical face crops cleanly
            (a face box must exceed 448 px for a 112 px output before the cap
            activates). Beyond the cap, mild residual aliasing may remain —
            because sample spacing D/S > 1 source pixel leaves some pixels
            uncovered — but the result is still far better than S=1.

    Returns:
        ``(N, C, H_out, W_out)`` crops.
    """
    C, H, W = img.shape
    N = boxes.shape[0]
    device, dtype = img.device, img.dtype

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    gx = (x1 + x2) / 2 / (W - 1) * 2 - 1
    gy = (y1 + y2) / 2 / (H - 1) * 2 - 1
    gw = (x2 - x1) / (W - 1)
    gh = (y2 - y1) / (H - 1)

    downsample = max(
        (x2 - x1).amax().item() / output_size[1],
        (y2 - y1).amax().item() / output_size[0],
    )
    S = int(min(max_supersample, max(1, math.ceil(downsample))))
    oh, ow = output_size[0] * S, output_size[1] * S

    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, oh, device=device, dtype=dtype),
        torch.linspace(-1, 1, ow, device=device, dtype=dtype),
        indexing="ij",
    )
    base = torch.stack([xs, ys], dim=-1).unsqueeze(0)  # (1, oh, ow, 2)
    grid_x = base[..., 0] * gw.view(N, 1, 1) + gx.view(N, 1, 1)
    grid_y = base[..., 1] * gh.view(N, 1, 1) + gy.view(N, 1, 1)
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (N, oh, ow, 2)

    img_batch = img.unsqueeze(0).expand(N, C, H, W)
    sampled = F.grid_sample(img_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    if S > 1:
        sampled = F.avg_pool2d(sampled, kernel_size=S)  # area filter -> (N, C, H_out, W_out)
    return sampled


def align(
    img: torch.Tensor,
    ldmks: torch.Tensor,
    template: torch.Tensor = ARCFACE_112_TEMPLATE,
    output_size: Tuple[int, int] = (112, 112),
    max_supersample: int = 4,
) -> torch.Tensor:
    """Differentiably align faces onto a template via a 5-point similarity fit.

    Args:
        img: ``(C, H, W)`` image (any positive range; range is preserved).
        ldmks: ``(F, 5, 2)`` detected landmarks in input-pixel coordinates.
        template: ``(5, 2)`` template landmarks in output-pixel coordinates.
            Defaults to :data:`ARCFACE_112_TEMPLATE`; use
            :func:`arcface_template` for other crop sizes.
        output_size: Output ``(H, W)``.
        max_supersample: Maximum anti-aliasing supersample factor.

            Anti-aliasing is achieved by evaluating the transform
            on an ``S``-times finer internal grid (``S²`` sub-pixel samples per
            output pixel) and then box-averaging them with ``avg_pool2d(kernel=S)``.
            Plain bilinear ``grid_sample`` at the output resolution reads only a
            2×2 source neighborhood per output pixel and skips source pixels in
            between, aliasing the crop. S is set to ``ceil(1 / transform_scale)``
            where ``transform_scale`` is the similarity transform's scale factor
            (output pixels per source pixel). This spaces sub-samples ≤ 1 source
            pixel apart so every source pixel contributes to the average, and the
            box average becomes an exact area filter over each output pixel's
            footprint in the source image. S = 1 is used when the transform
            upsamples (scale ≥ 1), which does not alias.

            This parameter caps S to bound memory: the intermediate tensor is
            ``(F, C, H_out×S, W_out×S)``, so S=8 costs 64× the memory of the
            final crop. The default of 4 handles all typical face crops cleanly
            (the transform scale must fall below 0.25, i.e. the face must span
            >448 px at a 112 px target, before the cap activates). Beyond the
            cap, mild residual aliasing may remain — because sample spacing
            1/(S·scale) > 1 source pixel leaves some pixels uncovered — but the
            result is still far better than S=1.

    Returns:
        ``(F, C, H_out, W_out)`` aligned crops, sampled from ``img``.
    """
    F_num = ldmks.shape[0]
    C, H, W = img.shape
    device, dtype = img.device, img.dtype
    oh, ow = output_size

    matrix = estimate_similarity_transform(ldmks, template)  # (F, 2, 3) src->dst
    inv = invert_similarity(matrix)                          # (F, 2, 3) dst->src

    scale = torch.sqrt(matrix[:, 0, 0] ** 2 + matrix[:, 1, 0] ** 2)  # (F,)
    downsample = (1.0 / scale.clamp_min(1e-6)).amax().item()
    S = int(min(max_supersample, max(1, math.ceil(downsample))))
    hs, ws = oh * S, ow * S

    # Sub-pixel output coordinates: S samples centered within each output pixel.
    ys, xs = torch.meshgrid(
        (torch.arange(hs, device=device, dtype=dtype) + 0.5) / S - 0.5,
        (torch.arange(ws, device=device, dtype=dtype) + 0.5) / S - 0.5,
        indexing="ij",
    )
    homog = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).reshape(-1, 3)  # (P, 3)
    src = inv @ homog.t()                                    # (F, 2, P)
    sx, sy = src[:, 0, :], src[:, 1, :]
    gx = 2.0 * sx / (W - 1) - 1.0
    gy = 2.0 * sy / (H - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).reshape(F_num, hs, ws, 2)

    img_batch = img.unsqueeze(0).expand(F_num, C, H, W)
    sampled = F.grid_sample(img_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    if S > 1:
        sampled = F.avg_pool2d(sampled, kernel_size=S)       # area filter -> (F, C, H_out, W_out)
    return sampled


def estimate_similarity_transform(landmarks: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Closed-form batched similarity transform mapping landmarks -> template.

    Fits, per face, the 4-DoF similarity ``[a, b, tx, ty]`` (uniform scale +
    rotation + translation)::

        x' = a * x - b * y + tx
        y' = b * x + a * y + ty

    in least squares from the 5 correspondences. Uses only reductions/products
    (no ``torch.linalg.lstsq``), so it is differentiable and MPS-safe.

    Args:
        landmarks: ``(F, 5, 2)`` source points (input pixels).
        template: ``(5, 2)`` destination points (output pixels).

    Returns:
        ``(F, 2, 3)`` affine matrices mapping source -> destination.
    """
    device, dtype = landmarks.device, landmarks.dtype
    dst = template.to(device=device, dtype=dtype)
    num = landmarks.shape[0]

    src_mean = landmarks.mean(dim=1)          # (F, 2)
    dst_mean = dst.mean(dim=0)                # (2,)
    src_c = landmarks - src_mean.unsqueeze(1)  # (F, 5, 2)
    dst_c = dst - dst_mean                    # (5, 2)

    sx, sy = src_c[..., 0], src_c[..., 1]     # (F, 5)
    dx, dy = dst_c[:, 0], dst_c[:, 1]         # (5,)
    denom = (sx * sx + sy * sy).sum(dim=1).clamp_min(1e-8)  # (F,)
    a = (sx * dx + sy * dy).sum(dim=1) / denom
    b = (sx * dy - sy * dx).sum(dim=1) / denom
    tx = dst_mean[0] - (a * src_mean[:, 0] - b * src_mean[:, 1])
    ty = dst_mean[1] - (b * src_mean[:, 0] + a * src_mean[:, 1])

    matrix = torch.zeros(num, 2, 3, device=device, dtype=dtype)
    matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2] = a, -b, tx
    matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2] = b, a, ty
    return matrix


def invert_similarity(matrix: torch.Tensor) -> torch.Tensor:
    """Analytically invert batched similarity transforms (MPS-safe).

    Args:
        matrix: ``(F, 2, 3)`` similarity transforms ``[[a,-b,tx],[b,a,ty]]``.

    Returns:
        ``(F, 2, 3)`` inverse transforms, without using ``torch.linalg.inv``.
    """
    a = matrix[:, 0, 0]
    b = matrix[:, 1, 0]
    tx = matrix[:, 0, 2]
    ty = matrix[:, 1, 2]
    det = (a * a + b * b).clamp_min(1e-8)
    ai, bi = a / det, b / det

    inv = torch.zeros_like(matrix)
    inv[:, 0, 0], inv[:, 0, 1], inv[:, 0, 2] = ai, bi, -(ai * tx + bi * ty)
    inv[:, 1, 0], inv[:, 1, 1], inv[:, 1, 2] = -bi, ai, (bi * tx - ai * ty)
    return inv
