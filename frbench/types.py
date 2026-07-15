"""Structured return types for the FR pipeline.

Both types are :class:`typing.NamedTuple` subclasses, so they support tuple
unpacking, ``_asdict()``, and positional or keyword construction in addition
to the helpers documented below.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Union

import torch


class FREmbedResult(NamedTuple):
    """Result of :meth:`frbench.FR.forward` / :meth:`frbench.FR.embed`.

    Attributes:
        embeddings: ``(M, D)`` face embeddings. Empty ``(0, D)`` when no faces
            were produced (never ``None``).
        indices: ``indices[i]`` is the source-image index of ``embeddings[i]``.
        crops: ``(M, 3, H, W)`` normalized crops fed to the backbone. Empty
            ``(0, 3, H, W)`` when no faces were produced (never ``None``).

    Examples:
        >>> result = fr(imgs, l2_normalize=True)
        >>> result.embeddings.shape                    # (M, 512)
        >>> embeddings, indices, crops = result        # tuple unpacking works
        >>> if result:                                 # truthy iff faces found
        ...     sim = result.embeddings @ result.embeddings.T
        >>> [i for i in result.indices if i == 0]      # faces from image 0
    """

    embeddings: torch.Tensor
    indices: List[int]
    crops: torch.Tensor

    @property
    def num_faces(self) -> int:
        """Number of face embeddings produced."""
        return self.embeddings.shape[0]

    def __bool__(self) -> bool:
        """True when at least one face embedding was produced."""
        return self.num_faces > 0


class FRDetectResult(NamedTuple):
    """Result of :meth:`frbench.FR.detect` / :meth:`frbench.FaceDetector.detect`.

    Attributes:
        detections: One entry per input image. Each entry is ``None`` (no face)
            or a ``(K, 16)`` tensor whose 16 columns are, in order:

            ====== =======================================================
            Column Meaning
            ====== =======================================================
            0-3    Box ``x1, y1, x2, y2`` in input-pixel coordinates.
            4      Background probability (``1 - face probability``).
            5      Face probability (the detection confidence score).
            6-15   Five landmarks ``x1, y1, ..., x5, y5`` in input pixels:
                   left eye, right eye, nose, left mouth corner,
                   right mouth corner.
            ====== =======================================================

    Use :attr:`boxes`, :attr:`scores`, and :attr:`landmarks` instead of
    indexing columns by hand. To build one from another detector's output
    (e.g. to feed ``FR(..., detections=...)``), use :meth:`from_landmarks`.

    Examples:
        >>> dets = detector.detect(imgs)
        >>> dets.boxes[0]          # (K, 4) boxes of image 0, or None
        >>> dets.scores[0]         # (K,) face probabilities, or None
        >>> dets.landmarks[0]      # (K, 5, 2) landmarks, or None
        >>> if dets:               # truthy iff any face in any image
        ...     result = fr(imgs, detections=dets)
    """

    detections: List[Optional[torch.Tensor]]

    @property
    def num_images(self) -> int:
        """Number of input images this result covers."""
        return len(self.detections)

    def __bool__(self) -> bool:
        """True when at least one face was detected in any image."""
        return any(d is not None and d.shape[0] > 0 for d in self.detections)

    @property
    def boxes(self) -> List[Optional[torch.Tensor]]:
        """Per-image ``(K, 4)`` boxes ``[x1, y1, x2, y2]`` (``None`` = no face)."""
        return [None if d is None else d[:, :4] for d in self.detections]

    @property
    def scores(self) -> List[Optional[torch.Tensor]]:
        """Per-image ``(K,)`` face probabilities (``None`` = no face)."""
        return [None if d is None else d[:, 5] for d in self.detections]

    @property
    def landmarks(self) -> List[Optional[torch.Tensor]]:
        """Per-image ``(K, 5, 2)`` landmarks in input pixels (``None`` = no face)."""
        return [None if d is None else d[:, 6:16].reshape(-1, 5, 2) for d in self.detections]

    @classmethod
    def from_landmarks(
        cls,
        landmarks: Union[torch.Tensor, Sequence[Optional[torch.Tensor]]],
        boxes: Optional[Union[torch.Tensor, Sequence[Optional[torch.Tensor]]]] = None,
        scores: Optional[Union[torch.Tensor, Sequence[Optional[torch.Tensor]]]] = None,
    ) -> "FRDetectResult":
        """Build an :class:`FRDetectResult` from third-party detector outputs.

        This is the supported way to feed landmarks from your own face
        detector into ``FR(..., detections=...)``.

        Args:
            landmarks: Per-image landmarks in input-pixel coordinates. Either a
                single ``(5, 2)`` / ``(K, 5, 2)`` tensor (one image) or a
                sequence with one entry per image, each ``None`` (no face) or a
                ``(5, 2)`` / ``(K, 5, 2)`` tensor. Landmark order is left eye,
                right eye, nose, left mouth corner, right mouth corner.
            boxes: Optional matching ``(K, 4)`` boxes ``[x1, y1, x2, y2]``
                per image. When omitted, the landmark bounding box is used
                (adequate for alignment and ``keep_largest`` selection).
            scores: Optional matching ``(K,)`` face probabilities per image.
                Defaults to ``1.0`` for every face.

        Returns:
            An :class:`FRDetectResult` with the standard 16-column layout.

        Example:
            >>> ldm = torch.tensor([[38., 52.], [74., 52.], [56., 72.],
            ...                     [42., 92.], [71., 92.]])          # (5, 2)
            >>> dets = FRDetectResult.from_landmarks([ldm, None])     # 2 images
            >>> result = fr([img0, img1], detections=dets)
        """
        def _as_list(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return [value]
            return list(value)

        ldm_list = _as_list(landmarks)
        box_list = _as_list(boxes)
        score_list = _as_list(scores)
        for name, lst in (("boxes", box_list), ("scores", score_list)):
            if lst is not None and len(lst) != len(ldm_list):
                raise ValueError(
                    f"Expected {len(ldm_list)} per-image {name} entries, got {len(lst)}"
                )

        detections: List[Optional[torch.Tensor]] = []
        for i, ldm in enumerate(ldm_list):
            if ldm is None:
                detections.append(None)
                continue
            if ldm.dim() == 2:
                ldm = ldm.unsqueeze(0)
            if ldm.dim() != 3 or ldm.shape[-2:] != (5, 2):
                raise ValueError(
                    f"landmarks[{i}] must have shape (5, 2) or (K, 5, 2), got {tuple(ldm.shape)}"
                )
            ldm = ldm.float()
            k = ldm.shape[0]

            box = box_list[i] if box_list is not None else None
            if box is None:
                box = torch.cat([ldm.amin(dim=1), ldm.amax(dim=1)], dim=-1)  # (K, 4)
            else:
                box = box.float().reshape(-1, 4)
                if box.shape[0] != k:
                    raise ValueError(
                        f"boxes[{i}] has {box.shape[0]} rows but landmarks[{i}] has {k} faces"
                    )
            box = box.to(ldm.device)

            score = score_list[i] if score_list is not None else None
            if score is None:
                score = torch.ones(k, device=ldm.device)
            else:
                score = score.float().reshape(-1).to(ldm.device)
                if score.shape[0] != k:
                    raise ValueError(
                        f"scores[{i}] has {score.shape[0]} entries but landmarks[{i}] has {k} faces"
                    )
            score = score.unsqueeze(-1)  # (K, 1)

            detections.append(
                torch.cat([box, 1.0 - score, score, ldm.reshape(k, 10)], dim=-1)
            )
        return cls(detections=detections)
