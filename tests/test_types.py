"""FRDetectResult / FREmbedResult accessors and builders (offline)."""
import pytest
import torch

from frbench import FRDetectResult, FREmbedResult


def _det_row(box, score, ldm):
    return torch.cat([
        torch.tensor(box, dtype=torch.float32),
        torch.tensor([1.0 - score, score]),
        torch.tensor(ldm, dtype=torch.float32).reshape(10),
    ])


LDM = [[38.0, 52.0], [74.0, 52.0], [56.0, 72.0], [42.0, 92.0], [71.0, 92.0]]


def test_detect_result_accessors():
    det = torch.stack([_det_row([10, 20, 110, 140], 0.99, LDM)])
    result = FRDetectResult(detections=[det, None])

    assert result.num_images == 2
    assert bool(result) is True
    assert result.boxes[0].tolist() == [[10, 20, 110, 140]]
    assert result.boxes[1] is None
    assert torch.allclose(result.scores[0], torch.tensor([0.99]))
    assert result.scores[1] is None
    assert result.landmarks[0].shape == (1, 5, 2)
    assert result.landmarks[0][0].tolist() == LDM
    assert result.landmarks[1] is None


def test_detect_result_falsy_when_empty():
    assert bool(FRDetectResult(detections=[None, None])) is False
    assert bool(FRDetectResult(detections=[torch.empty(0, 16)])) is False


def test_from_landmarks_defaults():
    ldm = torch.tensor(LDM)
    result = FRDetectResult.from_landmarks([ldm, None])

    assert result.num_images == 2
    det = result.detections[0]
    assert det.shape == (1, 16)
    assert result.detections[1] is None
    # Default box = landmark bounding box; default score = 1.0.
    assert det[0, :4].tolist() == [38.0, 52.0, 74.0, 92.0]
    assert det[0, 4].item() == pytest.approx(0.0)
    assert det[0, 5].item() == pytest.approx(1.0)
    assert result.landmarks[0][0].tolist() == LDM


def test_from_landmarks_single_tensor_and_explicit_fields():
    ldm = torch.tensor(LDM).unsqueeze(0).repeat(2, 1, 1)  # (2, 5, 2)
    boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [10.0, 10.0, 90.0, 90.0]])
    scores = torch.tensor([0.9, 0.8])
    result = FRDetectResult.from_landmarks(ldm, boxes=boxes, scores=scores)

    det = result.detections[0]
    assert det.shape == (2, 16)
    assert torch.equal(result.boxes[0], boxes)
    assert torch.allclose(result.scores[0], scores)
    assert torch.allclose(det[:, 4], 1.0 - scores)


def test_from_landmarks_roundtrip_through_accessors():
    det = torch.stack([_det_row([10, 20, 110, 140], 0.97, LDM)])
    native = FRDetectResult(detections=[det])
    rebuilt = FRDetectResult.from_landmarks(
        native.landmarks, boxes=native.boxes, scores=native.scores
    )
    assert torch.allclose(rebuilt.detections[0], native.detections[0], atol=1e-6)


def test_from_landmarks_validation():
    with pytest.raises(ValueError):
        FRDetectResult.from_landmarks([torch.zeros(3, 2)])  # not (K, 5, 2)
    with pytest.raises(ValueError):
        FRDetectResult.from_landmarks([torch.zeros(5, 2)], boxes=[None, None])
    with pytest.raises(ValueError):
        FRDetectResult.from_landmarks(
            [torch.zeros(2, 5, 2)], scores=[torch.tensor([0.5])]
        )


def test_embed_result_helpers():
    empty = FREmbedResult(
        embeddings=torch.empty(0, 512), indices=[], crops=torch.empty(0, 3, 112, 112)
    )
    assert empty.num_faces == 0
    assert bool(empty) is False

    filled = FREmbedResult(
        embeddings=torch.zeros(2, 512), indices=[0, 1], crops=torch.zeros(2, 3, 112, 112)
    )
    assert filled.num_faces == 2
    assert bool(filled) is True
    embeddings, indices, crops = filled  # tuple unpacking
    assert embeddings.shape == (2, 512) and indices == [0, 1]
