"""Standalone face detector with alignment and cropping conveniences."""
from __future__ import annotations

from numbers import Integral
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .types import FRDetectResult
from .utils.geometry import (
    align,
    arcface_template,
    crop,
    square_boxes,
    unalign,
    validate_input_range,
)
from .utils.retinaface import RetinaFace
from .utils.update_check import check_for_updates

#: ImageNet BGR channel means subtracted by RetinaFace at train/eval time.
RETINAFACE_BGR_MEAN = [104.0, 117.0, 123.0]

ImagesLike = Union[torch.Tensor, List[torch.Tensor]]
DetectionsLike = Union[FRDetectResult, Sequence[Optional[torch.Tensor]]]
AlignedImagesLike = Union[torch.Tensor, Sequence[torch.Tensor]]
OutputSizesLike = Union[Tuple[int, int], Sequence[Tuple[int, int]]]


class FaceDetector(nn.Module):
    """Standalone RetinaFace detector, independent of any :class:`frbench.FR`.

    Takes the same input contract as the rest of FRBench — RGB **float**
    tensors in ``[0, 255]`` — and handles RetinaFace's BGR mean-subtraction
    internally. Weights are downloaded lazily on first use.

    A single instance can be shared across multiple :class:`frbench.FR`
    pipelines via ``FR(..., detector=detector)`` so the detector weights are
    loaded only once. (A shared detector lives on one device: calling
    ``.to(device)`` on any pipeline holding it moves it for all of them.)

    Examples:
        >>> detector = frbench.FaceDetector().to(device)
        >>> dets = detector.detect(img)               # FRDetectResult
        >>> dets.boxes[0], dets.landmarks[0]          # (K, 4), (K, 5, 2)
        >>> aligned = detector.align(img, dets)[0]    # (K, 3, 112, 112), [0, 255]
        >>> restored = detector.unalign(aligned, dets, output_sizes=img.shape[-2:])
        >>> cropped = detector.crop(img, loosen=1.3)[0]
    """

    def __init__(
        self,
        name: str = "retinaface_mobilenetv1",
        *,
        eager_load: bool = False,
    ) -> None:
        """Initialize the detector.

        Args:
            name: RetinaFace asset name in the release manifest
                (``retinaface_mobilenetv1`` or ``retinaface_resnet50``).
            eager_load: If ``True``, download the weights and build the network
                immediately instead of on first use.
        """
        super().__init__()
        check_for_updates()
        self.name = name
        self.retinaface: Optional[RetinaFace] = None
        self.register_buffer(
            "bgr_mean",
            torch.tensor(RETINAFACE_BGR_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer("_dev", torch.empty(0), persistent=False)
        if eager_load:
            self._build()

    @property
    def device(self) -> torch.device:
        """Device this detector (and its lazily-built network) lives on."""
        return self._dev.device

    def _build(self) -> None:
        """Download the weights and build the RetinaFace network (once)."""
        if self.retinaface is None:
            self.retinaface = RetinaFace(self.name).to(self.device)
            self.retinaface.eval()

    def _as_image_list(self, imgs: ImagesLike) -> List[torch.Tensor]:
        """Normalize any supported input form to a list of ``(3, H, W)`` tensors."""
        if isinstance(imgs, (list, tuple)):
            out = []
            for img in imgs:
                if img.dim() == 4:
                    img = img.squeeze(0)
                if img.dim() != 3 or img.shape[0] != 3:
                    raise ValueError(
                        f"Expected (3,H,W) or (1,3,H,W) list items, got {tuple(img.shape)}"
                    )
                out.append(img)
            return out
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        if imgs.dim() != 4 or imgs.shape[1] != 3:
            raise ValueError(f"Expected (B,3,H,W) or (3,H,W), got {tuple(imgs.shape)}")
        return list(imgs.unbind(dim=0))

    @staticmethod
    def _as_aligned_list(aligned: AlignedImagesLike) -> List[torch.Tensor]:
        """Normalize aligned faces to one ``(K, C, H, W)`` tensor per source image."""
        groups = [aligned] if isinstance(aligned, torch.Tensor) else list(aligned)
        out: List[torch.Tensor] = []
        for i, group in enumerate(groups):
            if group.dim() == 3:
                group = group.unsqueeze(0)
            if group.dim() != 4:
                raise ValueError(
                    f"aligned[{i}] must have shape (C,H,W) or (K,C,H,W), "
                    f"got {tuple(group.shape)}"
                )
            out.append(group)
        return out

    @staticmethod
    def _resolve_output_sizes(
        output_sizes: OutputSizesLike,
        num_images: int,
    ) -> List[Tuple[int, int]]:
        """Normalize one shared or one-per-image source canvas size."""
        if (
            len(output_sizes) == 2
            and all(isinstance(value, Integral) for value in output_sizes)
        ):
            sizes = [(int(output_sizes[0]), int(output_sizes[1]))] * num_images
        else:
            sizes = [tuple(map(int, size)) for size in output_sizes]
            if len(sizes) != num_images:
                raise ValueError(f"Expected {num_images} output sizes, got {len(sizes)}")
        for i, size in enumerate(sizes):
            if len(size) != 2 or min(size) < 2:
                raise ValueError(
                    f"output_sizes[{i}] must be an (H,W) pair with both values >= 2"
                )
        return sizes

    def _detect_batch(
        self,
        imgs: torch.Tensor,
        conf_thresh: float,
        iou_thresh: float,
    ) -> List[Optional[torch.Tensor]]:
        """Detect on an already-batched ``(B, 3, H, W)`` tensor."""
        imgs = imgs.to(self.device).float()
        validate_input_range(imgs)
        self._build()
        with torch.no_grad():
            return self.retinaface(
                imgs[:, [2, 1, 0], :, :] - self.bgr_mean,
                need_decode=True,
                need_nms=True,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
            )

    def detect(
        self,
        imgs: ImagesLike,
        conf_thresh: float = 0.8,
        iou_thresh: float = 0.4,
    ) -> FRDetectResult:
        """Detect faces in one or more images.

        Args:
            imgs: RGB tensors in ``[0, 255]`` — ``(3,H,W)``, ``(B,3,H,W)``, or a
                list of per-image tensors (mixed sizes allowed).
            conf_thresh: Detector confidence threshold.
            iou_thresh: Detector NMS IoU threshold.

        Returns:
            :class:`FRDetectResult` with one detection tensor (or ``None``) per
            image; see its docs for the column layout and accessors.
        """
        if isinstance(imgs, (list, tuple)):
            all_dets: List[Optional[torch.Tensor]] = []
            for img in self._as_image_list(imgs):
                all_dets.extend(
                    self._detect_batch(img.unsqueeze(0), conf_thresh, iou_thresh)
                )
            return FRDetectResult(detections=all_dets)
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        if imgs.dim() != 4 or imgs.shape[1] != 3:
            raise ValueError(f"Expected (B,3,H,W) or (3,H,W), got {tuple(imgs.shape)}")
        return FRDetectResult(
            detections=self._detect_batch(imgs, conf_thresh, iou_thresh)
        )

    def _resolve_detections(
        self,
        images: List[torch.Tensor],
        detections: Optional[DetectionsLike],
        conf_thresh: float,
        iou_thresh: float,
    ) -> List[Optional[torch.Tensor]]:
        if detections is None:
            return self.detect(images, conf_thresh=conf_thresh, iou_thresh=iou_thresh).detections
        dets = (
            detections.detections
            if isinstance(detections, FRDetectResult)
            else list(detections)
        )
        if len(dets) != len(images):
            raise ValueError(f"Expected {len(images)} detections, got {len(dets)}")
        return dets

    @staticmethod
    def _keep_largest(det: torch.Tensor) -> torch.Tensor:
        areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]) * det[:, 5]
        return det[areas.argmax().unsqueeze(0)]

    def align(
        self,
        imgs: ImagesLike,
        detections: Optional[DetectionsLike] = None,
        *,
        size: Tuple[int, int] = (112, 112),
        template: Optional[torch.Tensor] = None,
        keep_largest: bool = False,
        conf_thresh: float = 0.8,
        iou_thresh: float = 0.4,
        max_supersample: int = 4,
    ) -> List[torch.Tensor]:
        """Detect (if needed) and 5-point align faces; crops stay in ``[0, 255]``.

        The alignment itself is :func:`frbench.align` and is differentiable
        w.r.t. the input images; only the detection step is not.

        Args:
            imgs: RGB tensors in ``[0, 255]`` — ``(3,H,W)``, ``(B,3,H,W)``, or a
                list of per-image tensors.
            detections: Optional precomputed detections (from :meth:`detect` or
                :meth:`FRDetectResult.from_landmarks`); skips re-detection.
            size: Output crop size ``(H, W)``.
            template: ``(5, 2)`` alignment template in output-pixel coordinates.
                Defaults to the ArcFace template scaled to ``size`` (see
                :func:`frbench.arcface_template`).
            keep_largest: Keep only the largest (area x confidence) face per image.
            conf_thresh: Detector confidence threshold (ignored when
                ``detections`` is given).
            iou_thresh: Detector NMS IoU threshold (ignored when ``detections``
                is given).
            max_supersample: Anti-aliasing cap; see :func:`frbench.align`.

        Returns:
            One ``(K, 3, H, W)`` tensor of un-normalized ``[0, 255]`` crops per
            image (``K`` may be 0 when no face was found).
        """
        images = self._as_image_list(imgs)
        dets = self._resolve_detections(images, detections, conf_thresh, iou_thresh)
        tmpl = template if template is not None else arcface_template(size)

        out: List[torch.Tensor] = []
        for img, det in zip(images, dets):
            img = img.to(self.device).float()
            if det is None or det.shape[0] == 0:
                out.append(img.new_zeros(0, 3, size[0], size[1]))
                continue
            if keep_largest:
                det = self._keep_largest(det)
            ldmks = det[:, 6:16].reshape(-1, 5, 2).to(img.device)
            out.append(align(img, ldmks, tmpl, size, max_supersample))
        return out

    def unalign(
        self,
        aligned: AlignedImagesLike,
        detections: DetectionsLike,
        *,
        output_sizes: OutputSizesLike,
        template: Optional[torch.Tensor] = None,
        keep_largest: bool = False,
        max_supersample: int = 4,
    ) -> List[torch.Tensor]:
        """Reproject aligned faces onto source-image-sized canvases.

        This reverses the spatial mapping used by :meth:`align`, but cannot
        recover source pixels outside an aligned crop. Those regions are
        zero-filled, and multiple faces remain on separate canvases.

        Args:
            aligned: Aligned faces, normally the list returned by :meth:`align`.
                Accepts one ``(K,C,H,W)`` tensor for one source image or a
                sequence with one tensor per source image. ``(C,H,W)`` is
                accepted as a one-face convenience.
            detections: The detections whose landmarks were used for alignment.
            output_sizes: Source canvas ``(H,W)`` shared by every image, or one
                ``(H,W)`` pair per source image.
            template: Alignment template used to create the faces. By default,
                the ArcFace template is scaled to each aligned tensor's size.
            keep_largest: Apply the same largest-face selection as :meth:`align`.
                Set this when the aligned inputs came from
                ``align(..., keep_largest=True)``.
            max_supersample: Anti-aliasing cap; see :func:`frbench.unalign`.

        Returns:
            One ``(K,C,H,W)`` tensor per source image. Each face occupies a
            separate, zero-padded source-coordinate canvas.
        """
        groups = self._as_aligned_list(aligned)
        dets = (
            detections.detections
            if isinstance(detections, FRDetectResult)
            else list(detections)
        )
        if len(dets) != len(groups):
            raise ValueError(f"Expected {len(groups)} detections, got {len(dets)}")
        sizes = self._resolve_output_sizes(output_sizes, len(groups))

        out: List[torch.Tensor] = []
        for i, (group, det, output_size) in enumerate(zip(groups, dets, sizes)):
            group = group.to(self.device).float()
            if det is None or det.shape[0] == 0:
                if group.shape[0] != 0:
                    raise ValueError(
                        f"aligned[{i}] has {group.shape[0]} faces but detections[{i}] "
                        "has none"
                    )
                out.append(group.new_empty((0, group.shape[1], *output_size)))
                continue
            if keep_largest:
                det = self._keep_largest(det)
            if group.shape[0] != det.shape[0]:
                raise ValueError(
                    f"aligned[{i}] has {group.shape[0]} faces but detections[{i}] "
                    f"has {det.shape[0]}"
                )
            ldmks = det[:, 6:16].reshape(-1, 5, 2)
            tmpl = (
                template
                if template is not None
                else arcface_template((group.shape[-2], group.shape[-1]))
            )
            out.append(
                unalign(
                    group,
                    ldmks,
                    output_size,
                    template=tmpl,
                    max_supersample=max_supersample,
                )
            )
        return out

    def crop(
        self,
        imgs: ImagesLike,
        detections: Optional[DetectionsLike] = None,
        *,
        size: Tuple[int, int] = (112, 112),
        loosen: float = 1.0,
        keep_largest: bool = False,
        conf_thresh: float = 0.8,
        iou_thresh: float = 0.4,
        max_supersample: int = 4,
    ) -> List[torch.Tensor]:
        """Detect (if needed) and crop square face boxes; crops stay in ``[0, 255]``.

        The cropping itself is :func:`frbench.crop` and is differentiable
        w.r.t. the input images; only the detection step is not.

        Args:
            imgs: RGB tensors in ``[0, 255]`` — ``(3,H,W)``, ``(B,3,H,W)``, or a
                list of per-image tensors.
            detections: Optional precomputed detections; skips re-detection.
            size: Output crop size ``(H, W)``.
            loosen: Enlargement factor applied to the squared detection box.
            keep_largest: Keep only the largest (area x confidence) face per image.
            conf_thresh: Detector confidence threshold (ignored when
                ``detections`` is given).
            iou_thresh: Detector NMS IoU threshold (ignored when ``detections``
                is given).
            max_supersample: Anti-aliasing cap; see :func:`frbench.crop`.

        Returns:
            One ``(K, 3, H, W)`` tensor of un-normalized ``[0, 255]`` crops per
            image (``K`` may be 0 when no face was found).
        """
        images = self._as_image_list(imgs)
        dets = self._resolve_detections(images, detections, conf_thresh, iou_thresh)

        out: List[torch.Tensor] = []
        for img, det in zip(images, dets):
            img = img.to(self.device).float()
            if det is None or det.shape[0] == 0:
                out.append(img.new_zeros(0, 3, size[0], size[1]))
                continue
            if keep_largest:
                det = self._keep_largest(det)
            boxes = square_boxes(det[:, :4].to(img.device), loosen)
            out.append(crop(img, boxes, size, max_supersample))
        return out
