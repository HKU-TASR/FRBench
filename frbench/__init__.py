import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from ._config import CACHE, RELEASE, REPO, __version__
from .fr import FR
from .types import FRDetectResult, FREmbedResult
from .utils.download import (
    ModelInfo,
    download_assets,
    get_asset,
    list_assets,
    list_models,
    refresh_manifest,
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
    "FREmbedResult",
    "FRDetectResult",
    "ModelInfo",
    "CACHE",
    "REPO",
    "RELEASE",
    "FRBenchWarning",
    "FRBenchError",
    "FRBenchDownloadError",
    "FRBenchAssetNotFoundError",
    "FRBenchConfigError",
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
