"""Structured return types for the FR pipeline."""
from __future__ import annotations

from typing import List, NamedTuple

import torch


class FREmbedResult(NamedTuple):
    """Result of :meth:`frbench.FR.forward` / :meth:`frbench.FR.embed`.

    Attributes:
        embeddings: ``(M, D)`` face embeddings. Empty ``(0, D)`` when no faces
            were produced (never ``None``).
        indices: ``indices[i]`` is the source-image index of ``embeddings[i]``.
        crops: ``(M, 3, H, W)`` normalized crops fed to the backbone. Empty
            ``(0, 3, H, W)`` when no faces were produced (never ``None``).
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
    """Result of :meth:`frbench.FR.detect`.

    Attributes:
        detections: One entry per input image. Each entry is ``None`` (no face)
            or a ``(K, 16)`` tensor with columns
            ``[box(4), cls_prob(2), ldm(10)]`` in pixel coordinates.
    """

    detections: List[torch.Tensor | None]
