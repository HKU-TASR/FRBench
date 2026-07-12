from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union
import os

import torch
from torch import nn
import yaml

from .backbones import build_backbone
from .types import FRDetectResult, FREmbedResult
from ._exceptions import FRBenchAssetNotFoundError
from .utils.download import get_asset, list_assets
from .utils.preprocess import Preprocessor
from .utils.postprocess import Postprocessor
from .utils.update_check import check_for_updates


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_asset_keys(model_name: str, detector_name: str) -> None:
    """Fail fast if model or detector keys are absent from the manifest."""
    assets = list_assets()
    if model_name not in assets:
        available = [a for a in assets if not a.startswith("retinaface_")]
        raise FRBenchAssetNotFoundError(
            f"Model '{model_name}' not in manifest. "
            f"Check (backbone, loss, dataset). Available models: {available}"
        )
    if detector_name not in assets:
        raise FRBenchAssetNotFoundError(
            f"Detector '{detector_name}' not in manifest. "
            f"Available detectors: {[a for a in assets if a.startswith('retinaface_')]}"
        )


class FR(nn.Module):
    """End-to-end face-recognition pipeline: detect -> align -> backbone -> embedding."""

    def __init__(
        self,
        backbone_type: str,
        loss_type: str,
        dataset_type: str,
        detector: str = "retinaface_mobilenetv1",
        *,
        eager_load: bool = False,
    ) -> None:
        """Initialize the pipeline.

        Args:
            backbone_type: Backbone key, e.g. ``irse-100`` or ``mobilefacenet``.
            loss_type: Loss key, e.g. ``arcface``.
            dataset_type: Training dataset key, e.g. ``ms1m``.
            detector: RetinaFace asset name for detection/alignment.
            eager_load: If ``True``, download weights and build the backbone and
                preprocessor immediately instead of on first use.
        """
        super(FR, self).__init__()
        check_for_updates()
        self.backbone_type = backbone_type
        self.loss_type = loss_type
        self.dataset_type = dataset_type
        self.model_name = f"{backbone_type}_{loss_type}_{dataset_type}"
        self.detector_name = detector

        _validate_asset_keys(self.model_name, self.detector_name)

        self.backbone: Optional[nn.Module] = None
        self.preprocessor: Optional[Preprocessor] = None
        self.postprocessor = Postprocessor()
        self.register_buffer("_dev", torch.empty(0), persistent=False)
        self._built = False
        self._embedding_dim: Optional[int] = None
        self._input_size: Tuple[int, int] = (112, 112)

        if eager_load:
            self._build()

    @property
    def device(self) -> torch.device:
        """Device this pipeline (and its lazily-built parts) lives on."""
        return self._dev.device

    @property
    def embedding_dim(self) -> int:
        """Output embedding dimension (requires the model to be built)."""
        if not self._built:
            self._build()
        assert self._embedding_dim is not None
        return self._embedding_dim

    @property
    def input_size(self) -> Tuple[int, int]:
        """Target crop size ``(H, W)`` read from the model config."""
        if not self._built:
            self._build()
        return self._input_size

    def _build(self) -> None:
        """Download the model and build the backbone + preprocessor (once)."""
        if self._built:
            return
        asset = get_asset(self.model_name)
        cfg = _load_yaml(os.path.join(asset["path"], asset["contents"]["config"]))
        bcfg = cfg["backbone"]
        icfg = cfg.get("input", {})

        self._embedding_dim = int(bcfg.get("kwargs", {}).get("num_features", 512))
        size = tuple(icfg.get("size", (112, 112)))
        self._input_size = (int(size[0]), int(size[1]))

        self.backbone = build_backbone(bcfg["name"], bcfg.get("kwargs", {}), self.device)
        state_dict = torch.load(
            os.path.join(asset["path"], asset["contents"]["weight"]),
            map_location="cpu",
            weights_only=True,
        )
        self.backbone.load_state_dict(state_dict, strict=True)
        self.backbone.eval()

        template = (
            torch.tensor(icfg["template"], dtype=torch.float32)
            if icfg.get("template") is not None
            else None
        )
        self.preprocessor = Preprocessor(
            detector_name=self.detector_name,
            mean=tuple(icfg.get("mean", (0.5, 0.5, 0.5))),
            std=tuple(icfg.get("var", icfg.get("std", (0.5, 0.5, 0.5)))),
            size=self._input_size,
            channel=icfg.get("channel", "rgb"),
            template=template,
        ).to(self.device)
        self._built = True

    def _empty_result(self) -> FREmbedResult:
        dim = self._embedding_dim or 512
        h, w = self._input_size
        device = self.device
        return FREmbedResult(
            embeddings=torch.empty(0, dim, device=device),
            indices=[],
            crops=torch.empty(0, 3, h, w, device=device),
        )

    def detect(
        self,
        imgs: Union[torch.Tensor, List[torch.Tensor]],
        conf_thresh: float = 0.8,
        iou_thresh: float = 0.4,
    ) -> FRDetectResult:
        """Run face detection without embedding (for detect-once attack workflows).

        Args:
            imgs: RGB tensors in ``[0, 255]`` — ``(3,H,W)``, ``(B,3,H,W)``, or a
                list of per-image tensors.
            conf_thresh: Detector confidence threshold.
            iou_thresh: Detector NMS IoU threshold.

        Returns:
            :class:`FRDetectResult` with one detection tensor (or ``None``) per image.
        """
        if not self._built:
            self._build()

        if isinstance(imgs, (list, tuple)):
            all_dets: List[Optional[torch.Tensor]] = []
            for img in imgs:
                if img.dim() == 4:
                    img = img.squeeze(0)
                dets = self.preprocessor.detect(
                    img.unsqueeze(0).to(self.device),
                    conf_thresh=conf_thresh,
                    iou_thresh=iou_thresh,
                )
                all_dets.extend(dets)
            return FRDetectResult(detections=all_dets)

        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        dets = self.preprocessor.detect(
            imgs.to(self.device),
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        )
        return FRDetectResult(detections=dets)

    def forward(
        self,
        imgs: Union[torch.Tensor, List[torch.Tensor]],
        need_crop: bool = True,
        need_align: bool = True,
        keep_largest: bool = True,
        discard_invalid: bool = False,
        conf_thresh: float = 0.8,
        iou_thresh: float = 0.4,
        loosen_crop: float = 1.0,
        tta: Sequence[str] = ("flip_horizontal",),
        l2_normalize: bool = False,
        detections: Optional[Union[FRDetectResult, Sequence[Optional[torch.Tensor]]]] = None,
    ) -> FREmbedResult:
        """Get embeddings for the given images.

        Args:
            imgs: Input images. Should be RGB and in the ``[0, 255]`` range. Can be:
                - A single image tensor of shape ``(3, H, W)``.
                - A batch of images of shape ``(B, 3, H, W)``.
                - A list of image tensors of shape ``(3, Hi, Wi)`` or ``(1, 3, Hi, Wi)``.
            need_crop: If ``False``, treat inputs as pre-cropped faces.
            need_align: If ``True`` (and ``need_crop``), 5-point align each face.
            keep_largest: Keep only the largest face per image.
            discard_invalid: Drop images with no detection.
            conf_thresh: Detector confidence threshold (ignored when ``detections`` set).
            iou_thresh: Detector NMS IoU threshold (ignored when ``detections`` set).
            loosen_crop: Box enlargement when ``need_align`` is ``False``.
            tta: Test-time augmentations to average over.
            l2_normalize: L2-normalize returned embeddings.
            detections: Precomputed detections from :meth:`detect` to skip
                re-detection during iterative optimization.

        Returns:
            :class:`FREmbedResult` with embeddings, source indices, and crops.
            Empty tensors (not ``None``) when no faces were produced.
        """
        if not self._built:
            self._build()

        precomputed: Optional[Sequence[Optional[torch.Tensor]]] = None
        if detections is not None:
            precomputed = (
                detections.detections
                if isinstance(detections, FRDetectResult)
                else detections
            )

        pre_kwargs = dict(
            need_crop=need_crop,
            need_align=need_align,
            keep_largest=keep_largest,
            discard_invalid=discard_invalid,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            loosen_crop=loosen_crop,
            detections=precomputed,
        )

        emb_chunks: List[torch.Tensor] = []
        crop_chunks: List[torch.Tensor] = []
        indices: List[int] = []

        if isinstance(imgs, (list, tuple)):
            group_crops: List[torch.Tensor] = []
            group_indices: List[int] = []
            det_offset = 0
            for offset, img in enumerate(imgs):
                img_dets = None
                if precomputed is not None:
                    img_dets = precomputed[det_offset : det_offset + 1]
                    det_offset += 1
                crops, _ = self.preprocessor(
                    img.to(self.device),
                    detections=img_dets,
                    **{k: v for k, v in pre_kwargs.items() if k != "detections"},
                )
                if crops is not None:
                    group_crops.append(crops)
                    group_indices.extend([offset] * crops.shape[0])
            if group_crops:
                crops = torch.cat(group_crops, dim=0)
                emb_chunks.append(
                    self.postprocessor(self.backbone, crops, tta=tta, l2_normalize=l2_normalize)
                )
                crop_chunks.append(crops)
                indices.extend(group_indices)
        else:
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(0)
            imgs = imgs.to(self.device)
            crops, local_indices = self.preprocessor(imgs, **pre_kwargs)
            if crops is not None:
                emb_chunks.append(
                    self.postprocessor(self.backbone, crops, tta=tta, l2_normalize=l2_normalize)
                )
                crop_chunks.append(crops)
                indices.extend(local_indices)

        if len(emb_chunks) == 0:
            return self._empty_result()
        return FREmbedResult(
            embeddings=torch.cat(emb_chunks, dim=0),
            indices=indices,
            crops=torch.cat(crop_chunks, dim=0),
        )

    def embed(
        self,
        imgs: Union[torch.Tensor, List[torch.Tensor]],
        **kwargs,
    ) -> FREmbedResult:
        """Alias for :meth:`forward` (invokes module hooks via ``__call__``)."""
        return self(imgs, **kwargs)
