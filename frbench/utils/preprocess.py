from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from ..detector import RETINAFACE_BGR_MEAN, FaceDetector
from .geometry import (
    ARCFACE_112_TEMPLATE,
    align,
    arcface_template,
    crop,
    estimate_similarity_transform,
    invert_similarity,
    square_boxes,
    validate_input_range,
)
from .log import warn

# Backward-compatible aliases (the implementations moved to .geometry / ..detector).
_square_boxes = square_boxes

__all__ = [
    "Preprocessor",
    "ARCFACE_112_TEMPLATE",
    "RETINAFACE_BGR_MEAN",
    "align",
    "arcface_template",
    "crop",
    "estimate_similarity_transform",
    "invert_similarity",
    "square_boxes",
]


class Preprocessor(nn.Module):
    """Detect faces and produce normalized, aligned crops for a backbone.
    """

    def __init__(
        self,
        detector: Union[str, FaceDetector] = "retinaface_mobilenetv1",
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        size: Tuple[int, int] = (112, 112),
        channel: str = "rgb",
        template: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            detector: RetinaFace asset name (lazily loaded on first crop) or an
                existing :class:`frbench.FaceDetector` instance to share.
            mean: Per-channel mean for backbone normalization (on a ``[0, 1]``
                scale, applied in the backbone's channel order).
            std: Per-channel std/divisor for backbone normalization. (The model
                configs name this ``var``; it is used directly as the divisor.)
            size: Target crop size ``(H, W)``.
            channel: Backbone channel order, ``"rgb"`` or ``"bgr"``.
            template: Optional ``(5, 2)`` alignment template in output-pixel
                coordinates. Defaults to :data:`ARCFACE_112_TEMPLATE`.
        """
        super().__init__()
        self.detector = detector if isinstance(detector, FaceDetector) else FaceDetector(detector)
        self.detector_name = self.detector.name
        self.size = tuple(size)
        self.channel = channel

        tmpl = template if template is not None else ARCFACE_112_TEMPLATE.clone()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("template", tmpl.to(torch.float32))
        # Zero-element marker buffer used only to track the module's device.
        self.register_buffer("_dev", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        """Device this module (and its lazily-built detector) lives on."""
        return self._dev.device

    def _normalize(self, faces: torch.Tensor) -> torch.Tensor:
        """Normalize ``[0, 255]`` RGB crops to the backbone's expected tensor.

        Args:
            faces: ``(N, 3, H, W)`` float tensor in ``[0, 255]``, RGB.

        Returns:
            ``(N, 3, H, W)`` normalized tensor in the configured channel order.
        """
        if self.channel == "bgr":
            faces = faces[:, [2, 1, 0], :, :]
        return (faces / 255.0 - self.mean) / self.std

    def detect(
        self,
        imgs: torch.Tensor,
        conf_thresh: float = 0.8,
        iou_thresh: float = 0.4,
    ) -> List[Optional[torch.Tensor]]:
        """Run RetinaFace detection without cropping or alignment.

        Returns one entry per image: ``None`` or a ``(K, 16)`` detection tensor.
        """
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        if imgs.dim() != 4 or imgs.shape[1] != 3:
            raise ValueError(f"Expected (B,3,H,W) or (3,H,W), got {tuple(imgs.shape)}")
        return self.detector.detect(
            imgs.to(self.device),
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        ).detections

    def forward(
        self,
        imgs: torch.Tensor,
        need_crop: bool = True,
        need_align: bool = True,
        keep_largest: bool = True,
        discard_invalid: bool = False,
        conf_thresh: float = 0.8,
        iou_thresh: float = 0.4,
        loosen_crop: float = 1.0,
        detections: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> Tuple[Optional[torch.Tensor], List[int]]:
        """Detect and preprocess faces.

        Args:
            imgs: ``(B, 3, H, W)`` or ``(3, H, W)`` RGB tensor in ``[0, 255]``.
            need_crop: If ``False``, treat inputs as pre-cropped faces (only
                resize + normalize) and skip detection.
            need_align: If ``True`` (and ``need_crop``), align via 5-point
                similarity transform; otherwise crop a (loosened) square box.
            keep_largest: Keep only the largest (area x confidence) face per
                image. If ``False``, keep every detected face.
            discard_invalid: If ``True``, images with no detection are dropped;
                otherwise the whole (resized) image is used as a fallback.
            conf_thresh: Detector confidence threshold.
            iou_thresh: Detector NMS IoU threshold.
            loosen_crop: Box enlargement factor when ``need_align`` is ``False``.
            detections: Optional precomputed detections (one per image). When
                provided, detection is skipped and these tensors are used
                directly (useful for iterative adversarial attacks).

        Returns:
            Tuple ``(faces, indices)`` where ``faces`` is an ``(M, 3, H, W)``
            normalized tensor (or ``None`` if nothing was produced) and
            ``indices[i]`` is the source-image index of ``faces[i]``.
        """
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        if imgs.dim() != 4 or imgs.shape[1] != 3:
            raise ValueError(f"Expected (B,3,H,W) or (3,H,W), got {tuple(imgs.shape)}")
        imgs = imgs.to(self.device).float()
        validate_input_range(imgs)

        if not need_crop:
            x = imgs
            if tuple(x.shape[-2:]) != self.size:
                # antialias=True low-pass filters before downsampling (no-op when upsampling).
                x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False, antialias=True)
            return self._normalize(x), list(range(imgs.shape[0]))

        if detections is None:
            dets = self.detect(imgs, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        else:
            if len(detections) != imgs.shape[0]:
                raise ValueError(
                    f"Expected {imgs.shape[0]} precomputed detections, got {len(detections)}"
                )
            dets = list(detections)

        faces255: List[torch.Tensor] = []
        indices: List[int] = []
        for b in range(imgs.shape[0]):
            det = dets[b]
            if det is None or det.shape[0] == 0:
                if not discard_invalid:
                    fallback = F.interpolate(imgs[[b]], size=self.size, mode="bilinear", align_corners=False, antialias=True)
                    faces255.append(fallback)
                    indices.append(b)
                    warn(f"[Preprocessor] no detection for image {b}; using full image.")
                else:
                    warn(f"[Preprocessor] no detection for image {b}; discarded.")
                continue

            if keep_largest:
                areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]) * det[:, 5]
                det = det[areas.argmax().unsqueeze(0)]

            if need_align:
                ldmks = det[:, 6:16].reshape(-1, 5, 2)
                crops = align(imgs[b], ldmks, self.template, self.size)
            else:
                boxes = square_boxes(det[:, :4], loosen_crop)
                crops = crop(imgs[b], boxes, self.size)
            faces255.append(crops)
            indices.extend([b] * crops.shape[0])

        if len(faces255) == 0:
            warn("[Preprocessor] no faces produced; returning None.")
            return None, []
        return self._normalize(torch.cat(faces255, dim=0)), indices
