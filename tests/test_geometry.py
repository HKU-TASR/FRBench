"""Alignment/cropping math on synthetic data (offline)."""
import math

import pytest
import torch

import frbench
from frbench.utils.geometry import square_boxes


def make_transform(scale, angle_deg, tx, ty):
    """Build one (2, 3) similarity matrix [[a,-b,tx],[b,a,ty]]."""
    theta = math.radians(angle_deg)
    a = scale * math.cos(theta)
    b = scale * math.sin(theta)
    return torch.tensor([[a, -b, tx], [b, a, ty]], dtype=torch.float32)


def apply_transform(matrix, points):
    """Apply (2, 3) affine to (N, 2) points."""
    homog = torch.cat([points, torch.ones(points.shape[0], 1)], dim=-1)
    return homog @ matrix.t()


def test_arcface_template_scaling():
    base = frbench.arcface_template((112, 112))
    assert torch.allclose(base, frbench.ARCFACE_112_TEMPLATE)
    double = frbench.arcface_template((224, 224))
    assert torch.allclose(double, frbench.ARCFACE_112_TEMPLATE * 2)
    wide = frbench.arcface_template((112, 224))  # (H, W): only x scales
    assert torch.allclose(wide[:, 0], frbench.ARCFACE_112_TEMPLATE[:, 0] * 2)
    assert torch.allclose(wide[:, 1], frbench.ARCFACE_112_TEMPLATE[:, 1])


def test_estimate_similarity_recovers_known_transform():
    template = frbench.ARCFACE_112_TEMPLATE
    true = make_transform(scale=0.5, angle_deg=20.0, tx=30.0, ty=-10.0)
    # Landmarks such that landmarks -> template under `true`: apply the inverse.
    inv = frbench.invert_similarity(true.unsqueeze(0))[0]
    landmarks = apply_transform(inv, template).unsqueeze(0)  # (1, 5, 2)

    est = frbench.estimate_similarity_transform(landmarks, template)[0]
    assert torch.allclose(est, true, atol=1e-3)


def test_invert_similarity_composes_to_identity():
    matrix = make_transform(scale=1.7, angle_deg=-33.0, tx=5.0, ty=12.0).unsqueeze(0)
    inv = frbench.invert_similarity(matrix)
    points = torch.rand(10, 2) * 100
    roundtrip = apply_transform(inv[0], apply_transform(matrix[0], points))
    assert torch.allclose(roundtrip, points, atol=1e-4)


def test_crop_extracts_expected_region():
    # Image with a constant-color 40x40 block; crop exactly that block.
    img = torch.zeros(3, 100, 100)
    img[:, 30:70, 20:60] = 200.0
    crops = frbench.crop(img, torch.tensor([[20.0, 30.0, 60.0, 40.0 + 30.0]]), (56, 56))
    assert crops.shape == (1, 3, 56, 56)
    # Interior of the crop is the block color (edges may blend).
    assert torch.allclose(crops[0, :, 5:-5, 5:-5], torch.full((3, 46, 46), 200.0), atol=1.0)


def test_crop_is_differentiable():
    img = (torch.rand(3, 64, 64) * 255).requires_grad_(True)
    crops = frbench.crop(img, torch.tensor([[8.0, 8.0, 40.0, 40.0]]), (16, 16))
    crops.sum().backward()
    assert img.grad is not None and img.grad.abs().sum() > 0


def test_align_identity_when_landmarks_match_template():
    # If landmarks already equal the template, alignment is (nearly) identity.
    img = torch.rand(3, 112, 112) * 255
    ldmks = frbench.ARCFACE_112_TEMPLATE.unsqueeze(0)
    aligned = frbench.align(img, ldmks, frbench.ARCFACE_112_TEMPLATE, (112, 112))
    assert aligned.shape == (1, 3, 112, 112)
    # float32 grid_sample on [0, 255] values: allow small interpolation error.
    assert torch.allclose(aligned[0], img, atol=1e-2)


def test_align_is_differentiable():
    img = (torch.rand(3, 200, 200) * 255).requires_grad_(True)
    ldmks = (frbench.ARCFACE_112_TEMPLATE + 40.0).unsqueeze(0)
    aligned = frbench.align(img, ldmks)
    aligned.sum().backward()
    assert img.grad is not None and img.grad.abs().sum() > 0


def test_unalign_identity_when_landmarks_match_template():
    aligned = torch.rand(2, 3, 112, 112) * 255
    ldmks = frbench.ARCFACE_112_TEMPLATE.unsqueeze(0).expand(2, -1, -1)
    restored = frbench.unalign(aligned, ldmks, (112, 112))
    assert restored.shape == aligned.shape
    assert torch.allclose(restored, aligned, atol=1e-2)


def test_unalign_uses_source_to_aligned_direction_and_zero_fills():
    aligned = torch.full((1, 3, 112, 112), 200.0)
    offset = torch.tensor([15.0, 20.0])
    ldmks = (frbench.ARCFACE_112_TEMPLATE + offset).unsqueeze(0)
    restored = frbench.unalign(aligned, ldmks, (150, 160))

    # The aligned crop is translated to x=15..126, y=20..131 in source space.
    assert torch.allclose(
        restored[0, :, 40:100, 35:95],
        torch.full((3, 60, 60), 200.0),
        atol=1e-3,
    )
    assert restored[0, :, :10].count_nonzero() == 0
    assert restored[0, :, :, :5].count_nonzero() == 0


def test_align_unalign_roundtrip_preserves_available_interior():
    ys, xs = torch.meshgrid(torch.arange(160), torch.arange(160), indexing="ij")
    img = torch.stack([xs, ys, xs + ys]).float()
    offset = torch.tensor([20.0, 30.0])
    ldmks = (frbench.ARCFACE_112_TEMPLATE + offset).unsqueeze(0)

    aligned = frbench.align(img, ldmks)
    restored = frbench.unalign(aligned, ldmks, (160, 160))
    assert torch.allclose(restored[0, :, 35:130, 25:120], img[:, 35:130, 25:120], atol=1e-3)
    assert restored[0, :, :20].count_nonzero() == 0


def test_unalign_keeps_multiple_faces_on_independent_canvases():
    aligned = torch.stack(
        [
            torch.full((3, 112, 112), 50.0),
            torch.full((3, 112, 112), 150.0),
        ]
    )
    offsets = torch.tensor([[10.0, 15.0], [30.0, 35.0]])
    ldmks = frbench.ARCFACE_112_TEMPLATE.unsqueeze(0) + offsets[:, None, :]
    restored = frbench.unalign(aligned, ldmks, (160, 170))

    assert restored.shape == (2, 3, 160, 170)
    assert restored[0, :, 50, 50].mean() == pytest.approx(50.0)
    assert restored[1, :, 70, 70].mean() == pytest.approx(150.0)
    assert restored[1, :, 20, 20].count_nonzero() == 0


def test_unalign_empty_batch_has_source_canvas_shape():
    aligned = torch.empty(0, 3, 112, 112)
    ldmks = torch.empty(0, 5, 2)
    restored = frbench.unalign(aligned, ldmks, (180, 240))
    assert restored.shape == (0, 3, 180, 240)


def test_unalign_antialiases_when_downsampling():
    ys, xs = torch.meshgrid(torch.arange(112), torch.arange(112), indexing="ij")
    checker = ((xs + ys) % 2).float().expand(3, -1, -1).unsqueeze(0)
    # Source landmarks at half the template coordinates produce a 2x
    # source-to-aligned scale, so reprojection downsamples 112 -> 56.
    ldmks = (frbench.ARCFACE_112_TEMPLATE / 2).unsqueeze(0)
    restored = frbench.unalign(
        checker,
        ldmks,
        (56, 56),
        max_supersample=2,
    )
    interior = restored[0, :, 2:-2, 2:-2]
    assert interior.mean() == pytest.approx(0.5, abs=1e-3)
    assert interior.std() < 1e-3


def test_unalign_is_differentiable():
    aligned = torch.rand(1, 3, 112, 112, requires_grad=True)
    ldmks = (frbench.ARCFACE_112_TEMPLATE + 20.0).unsqueeze(0).requires_grad_(True)
    restored = frbench.unalign(aligned, ldmks, (150, 150))
    restored.square().mean().backward()
    assert aligned.grad is not None and aligned.grad.abs().sum() > 0
    assert ldmks.grad is not None and ldmks.grad.abs().sum() > 0


def test_unalign_rejects_degenerate_landmarks():
    aligned = torch.rand(1, 3, 112, 112)
    with pytest.raises(ValueError, match="degenerate"):
        frbench.unalign(aligned, torch.ones(1, 5, 2), (112, 112))


def test_square_boxes():
    boxes = torch.tensor([[0.0, 0.0, 40.0, 20.0]])
    squared = square_boxes(boxes)
    x1, y1, x2, y2 = squared[0].tolist()
    assert x2 - x1 == pytest.approx(40.0)
    assert y2 - y1 == pytest.approx(40.0)
    assert (x1 + x2) / 2 == pytest.approx(20.0)
    assert (y1 + y2) / 2 == pytest.approx(10.0)
    loosened = square_boxes(boxes, loosen=1.5)
    assert loosened[0, 2] - loosened[0, 0] == pytest.approx(60.0)
