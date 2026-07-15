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
