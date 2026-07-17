import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from . import _config
from ._config import __version__, configure, configure_scoped
from .detector import FaceDetector
from .fr import FR
from .types import FRDetectResult, FREmbedResult, FRUnalignResult
from .utils.download import (
    ModelInfo,
    download_assets,
    get_asset,
    list_assets,
    list_models,
    refresh_manifest,
)
from .utils.geometry import (
    ARCFACE_112_TEMPLATE,
    align,
    arcface_template,
    crop,
    estimate_similarity_transform,
    invert_similarity,
    unalign,
)
from .utils.log import (
    FRBenchWarning,
    download_verbose_enabled,
    set_download_verbose,
    set_verbose,
    set_warnings,
    warnings_enabled,
)
from .utils.update_check import set_update_check, update_check_enabled
from ._exceptions import (
    FRBenchAssetNotFoundError,
    FRBenchConfigError,
    FRBenchDownloadError,
    FRBenchError,
)

__all__ = [
    "__version__",
    "FR",
    "FaceDetector",
    "FREmbedResult",
    "FRDetectResult",
    "FRUnalignResult",
    "ModelInfo",
    "CACHE",
    "REPO",
    "RELEASE",
    "ARCFACE_112_TEMPLATE",
    "align",
    "arcface_template",
    "crop",
    "estimate_similarity_transform",
    "invert_similarity",
    "unalign",
    "FRBenchWarning",
    "FRBenchError",
    "FRBenchDownloadError",
    "FRBenchAssetNotFoundError",
    "FRBenchConfigError",
    "configure",
    "configure_scoped",
    "set_warnings",
    "set_download_verbose",
    "set_verbose",
    "warnings_enabled",
    "download_verbose_enabled",
    "set_update_check",
    "update_check_enabled",
    "get_asset",
    "list_assets",
    "list_models",
    "download_assets",
    "refresh_manifest",
]


def __getattr__(name: str):
    """Expose ``CACHE`` / ``REPO`` / ``RELEASE`` as live, read-only views.

    These reflect the currently resolved configuration (including scoped
    overrides); change them with :func:`configure` or environment variables.
    """
    if name in ("CACHE", "REPO", "RELEASE"):
        return getattr(_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
