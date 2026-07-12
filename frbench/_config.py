"""Runtime configuration for FRBench (env vars + module-level overrides)."""
from __future__ import annotations

import os
from typing import Optional

__version__ = "1.0.0"

_DEFAULT_CACHE = os.path.expanduser(os.path.expandvars("~/.frbench"))
_DEFAULT_REPO = "HKU-TASR/FRBench"
_DEFAULT_RELEASE = "weights-v1.0.0"

# Module-level overrides (set before first download, or via env vars).
CACHE: str = os.environ.get("FRBENCH_CACHE", _DEFAULT_CACHE)
REPO: str = os.environ.get("FRBENCH_REPO", _DEFAULT_REPO)
RELEASE: str = os.environ.get("FRBENCH_RELEASE", _DEFAULT_RELEASE)

_DETECTOR_PREFIX = "retinaface_"


def get_cache() -> str:
    """Return the active cache directory."""
    return os.path.expanduser(os.path.expandvars(os.environ.get("FRBENCH_CACHE", CACHE)))


def get_repo() -> str:
    """Return the active GitHub repo slug."""
    return os.environ.get("FRBENCH_REPO", REPO)


def get_release() -> str:
    """Return the active GitHub release tag."""
    return os.environ.get("FRBENCH_RELEASE", RELEASE)


def is_detector_asset(name: str) -> bool:
    """True if *name* is a RetinaFace detector asset (not an FR model)."""
    return name.startswith(_DETECTOR_PREFIX)


def parse_model_key(name: str) -> Optional[tuple[str, str, str]]:
    """Parse a manifest model key into ``(backbone, loss, dataset)``.

    Returns ``None`` for detector assets or malformed keys.
    """
    if is_detector_asset(name):
        return None
    parts = name.rsplit("_", 2)
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]
