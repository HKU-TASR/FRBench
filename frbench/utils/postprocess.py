from typing import Sequence

import torch
from torch import nn

from .log import warn


SUPPORTED_TTA = ("flip_horizontal",)

class Postprocessor(nn.Module):
    """Apply optional TTA and produce final embeddings.
    """

    def _views(self, crops: torch.Tensor, tta: Sequence[str]) -> list:
        """Build the list of (original + augmented) views of the crops."""
        views = [crops]
        if "flip_horizontal" in tta:
            views.append(torch.flip(crops, dims=[-1]))
        return views

    def forward(
        self,
        backbone: nn.Module,
        crops: torch.Tensor,
        tta: Sequence[str] = (),
        l2_normalize: bool = False,
    ) -> torch.Tensor:
        """Embed crops, averaging over TTA views.

        Args:
            backbone: The embedding network mapping ``(N, 3, H, W) -> (N, D)``.
            crops: ``(M, 3, H, W)`` normalized face crops.
            tta: Augmentations to include in addition to the original view.
                Currently supports ``"flip_horizontal"``. Empty = no TTA.
            l2_normalize: If ``True``, L2-normalize embeddings (handy for direct
                cosine comparison). Defaults to ``False`` (raw embeddings).

        Returns:
            ``(M, D)`` embeddings (averaged over views; L2-normalized if
            ``l2_normalize``).
        """
        valid_tta = []
        for aug in tta:
            if aug not in SUPPORTED_TTA:
                warn(f"[Postprocessor] TTA '{aug}' is not supported. Skipped. Supported TTA: {SUPPORTED_TTA}")
            else:
                valid_tta.append(aug)
        views = self._views(crops, tuple(valid_tta))
        num_views, num_faces = len(views), crops.shape[0]
        # Single batched backbone call over all views for speed.
        embeddings = backbone(torch.cat(views, dim=0))
        embeddings = embeddings.view(num_views, num_faces, -1).mean(dim=0)
        if l2_normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings
