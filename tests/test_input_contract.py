"""Input validation and warning behavior, without downloading any weights."""
import pytest
import torch

import frbench
from frbench.detector import FaceDetector
from frbench.utils.geometry import validate_input_range
from frbench.utils.log import FRBenchWarning
from frbench.utils.preprocess import Preprocessor


def test_validate_input_range_raises_out_of_range():
    with pytest.raises(ValueError):
        validate_input_range(torch.full((1, 3, 8, 8), 300.0))
    with pytest.raises(ValueError):
        validate_input_range(torch.full((1, 3, 8, 8), -5.0))


def test_validate_input_range_warns_on_normalized_looking_input():
    with pytest.warns(FRBenchWarning):
        validate_input_range(torch.rand(1, 3, 8, 8))  # looks like [0, 1]


def test_validate_input_warning_respects_switch():
    import warnings as w

    frbench.set_warnings(False)
    with w.catch_warnings():
        w.simplefilter("error")  # any warning would fail the test
        validate_input_range(torch.rand(1, 3, 8, 8))
    frbench.set_warnings(True)


def test_detector_rejects_bad_shapes_before_building():
    detector = FaceDetector()  # no weights downloaded
    with pytest.raises(ValueError):
        detector.detect(torch.zeros(1, 1, 8, 8))  # wrong channel count
    with pytest.raises(ValueError):
        detector.detect(torch.zeros(8, 8))  # missing channel dim
    with pytest.raises(ValueError):
        detector.detect([torch.zeros(1, 8, 8)])  # bad list item
    assert detector.retinaface is None  # validation happened before any build


def test_detector_align_rejects_mismatched_detections():
    detector = FaceDetector()
    dets = frbench.FRDetectResult(detections=[None, None])
    with pytest.raises(ValueError):
        detector.align(torch.rand(3, 32, 32) * 255, dets)  # 1 image, 2 detections


def test_detector_align_empty_detections_yield_empty_crops():
    detector = FaceDetector()
    dets = frbench.FRDetectResult(detections=[None])
    aligned = detector.align(torch.rand(3, 32, 32) * 255, dets)
    cropped = detector.crop(torch.rand(3, 32, 32) * 255, dets)
    assert aligned[0].shape == (0, 3, 112, 112)
    assert cropped[0].shape == (0, 3, 112, 112)
    assert detector.retinaface is None  # never needed the network


def test_detector_unalign_supports_per_image_output_sizes():
    detector = FaceDetector()
    template = frbench.ARCFACE_112_TEMPLATE
    dets = frbench.FRDetectResult.from_landmarks(
        [template + torch.tensor([10.0, 15.0]), template + torch.tensor([20.0, 25.0])]
    )
    aligned = [
        torch.full((1, 3, 112, 112), 50.0),
        torch.full((1, 3, 112, 112), 150.0),
    ]

    restored = detector.unalign(
        aligned,
        dets,
        output_sizes=[(140, 150), (160, 180)],
    )
    assert [tuple(img.shape) for img in restored] == [
        (1, 3, 140, 150),
        (1, 3, 160, 180),
    ]
    assert restored[0][0, :, 50, 50].mean() == pytest.approx(50.0)
    assert restored[1][0, :, 60, 60].mean() == pytest.approx(150.0)
    assert detector.retinaface is None


def test_detector_unalign_mask_result_matches_per_image_structure():
    detector = FaceDetector()
    template = frbench.ARCFACE_112_TEMPLATE
    dets = frbench.FRDetectResult.from_landmarks(
        [
            torch.stack([template + 5.0, template + 15.0]),
            template + torch.tensor([20.0, 25.0]),
        ]
    )
    aligned = [
        torch.zeros(2, 3, 112, 112),
        torch.zeros(1, 3, 112, 112),
    ]

    result = detector.unalign(
        aligned,
        dets,
        output_sizes=[(140, 150), (160, 180)],
        return_mask=True,
    )
    assert isinstance(result, frbench.FRUnalignResult)
    canvases, masks = result
    assert [tuple(item.shape) for item in canvases] == [
        (2, 3, 140, 150),
        (1, 3, 160, 180),
    ]
    assert [tuple(item.shape) for item in masks] == [
        (2, 1, 140, 150),
        (1, 1, 160, 180),
    ]
    assert all(mask.dtype == torch.bool for mask in masks)
    assert all(mask.any() for mask in masks)
    assert all(canvas.count_nonzero() == 0 for canvas in canvases)


def test_detector_unalign_empty_detections_yield_empty_canvases():
    detector = FaceDetector()
    dets = frbench.FRDetectResult(detections=[None, torch.empty(0, 16)])
    restored = detector.unalign(
        [torch.empty(0, 3, 112, 112), torch.empty(0, 3, 112, 112)],
        dets,
        output_sizes=[(120, 130), (140, 150)],
    )
    assert restored[0].shape == (0, 3, 120, 130)
    assert restored[1].shape == (0, 3, 140, 150)

    result = detector.unalign(
        [torch.empty(0, 3, 112, 112), torch.empty(0, 3, 112, 112)],
        dets,
        output_sizes=[(120, 130), (140, 150)],
        return_mask=True,
    )
    assert result.canvases[0].shape == (0, 3, 120, 130)
    assert result.masks[0].shape == (0, 1, 120, 130)
    assert result.masks[1].shape == (0, 1, 140, 150)
    assert result.masks[0].dtype == torch.bool


def test_detector_unalign_validates_group_counts_and_sizes():
    detector = FaceDetector()
    template = frbench.ARCFACE_112_TEMPLATE
    one_face = frbench.FRDetectResult.from_landmarks(template)
    two_faces = frbench.FRDetectResult.from_landmarks(
        torch.stack([template, template + 10.0])
    )

    with pytest.raises(ValueError, match="detections"):
        detector.unalign(
            [torch.rand(1, 3, 112, 112), torch.rand(1, 3, 112, 112)],
            one_face,
            output_sizes=(140, 140),
        )
    with pytest.raises(ValueError, match=r"aligned\[0\]"):
        detector.unalign(
            torch.rand(1, 3, 112, 112),
            two_faces,
            output_sizes=(140, 140),
        )
    with pytest.raises(ValueError, match="output sizes"):
        detector.unalign(
            [torch.rand(1, 3, 112, 112), torch.rand(1, 3, 112, 112)],
            frbench.FRDetectResult(detections=one_face.detections * 2),
            output_sizes=[(140, 140)],
        )


def test_detector_unalign_matches_align_keep_largest_selection():
    detector = FaceDetector()
    template = frbench.ARCFACE_112_TEMPLATE
    landmarks = torch.stack([template + 5.0, template + 25.0])
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 40.0, 40.0]])
    dets = frbench.FRDetectResult.from_landmarks(landmarks, boxes=boxes)
    aligned = torch.full((1, 3, 112, 112), 100.0)

    restored = detector.unalign(
        aligned,
        dets,
        output_sizes=(160, 160),
        keep_largest=True,
    )
    assert restored[0].shape == (1, 3, 160, 160)
    assert restored[0][0, :, 60, 60].mean() == pytest.approx(100.0)
    assert restored[0][0, :, 10, 10].count_nonzero() == 0

    result = detector.unalign(
        aligned,
        dets,
        output_sizes=(160, 160),
        keep_largest=True,
        return_mask=True,
    )
    assert result.canvases[0].shape == (1, 3, 160, 160)
    assert result.masks[0].shape == (1, 1, 160, 160)
    assert result.masks[0][0, 0, 60, 60]
    assert not result.masks[0][0, 0, 10, 10]


def test_preprocessor_precropped_path_needs_no_detector():
    pre = Preprocessor()
    faces, indices = pre(torch.rand(2, 3, 112, 112) * 255, need_crop=False)
    assert faces.shape == (2, 3, 112, 112)
    assert indices == [0, 1]
    # Default normalization maps [0, 255] to [-1, 1].
    assert faces.min() >= -1.0001 and faces.max() <= 1.0001
    assert pre.detector.retinaface is None


def test_preprocessor_rejects_bad_shapes():
    pre = Preprocessor()
    with pytest.raises(ValueError):
        pre(torch.zeros(1, 4, 16, 16) + 100.0)
    with pytest.raises(ValueError):
        pre(torch.zeros(1, 1, 3, 16, 16) + 100.0)


def test_preprocessor_precomputed_detections_count_mismatch():
    pre = Preprocessor()
    with pytest.raises(ValueError):
        pre(torch.rand(2, 3, 64, 64) * 255, detections=[None])


def test_preprocessor_shares_given_detector_instance():
    detector = FaceDetector()
    pre = Preprocessor(detector=detector)
    assert pre.detector is detector
    assert pre.detector_name == detector.name
