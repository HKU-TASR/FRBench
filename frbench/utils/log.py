"""Verbosity control, diagnostics, and download progress.

This module is the single source of truth for the package's user-facing output.
There are two independent, on-the-fly switches (both enabled by default):

- Inference warnings (e.g. no face detected, unsupported TTA) go through the
  stdlib :mod:`warnings` module under the :class:`FRBenchWarning` category.
- Download progress/status (log messages + the ``tqdm`` bar) is a separate
  channel toggled with :func:`set_download_verbose`.
"""
from __future__ import annotations

import logging
from typing import Optional
import warnings

LOGGER = logging.getLogger("frbench")
LOGGER.addHandler(logging.NullHandler())

class FRBenchWarning(UserWarning):
    """Category for FRBench inference-time warnings (filterable via ``warnings``)."""


_WARNINGS = True
_DOWNLOAD_VERBOSE = True


def set_warnings(enabled: bool = True) -> None:
    """Enable or disable FRBench inference warnings."""
    global _WARNINGS
    _WARNINGS = bool(enabled)
    warnings.filterwarnings("always" if _WARNINGS else "ignore", category=FRBenchWarning)


def set_download_verbose(enabled: bool = True) -> None:
    """Enable or disable download status messages and progress bars."""
    global _DOWNLOAD_VERBOSE
    _DOWNLOAD_VERBOSE = bool(enabled)
    if _DOWNLOAD_VERBOSE:
        if not LOGGER.handlers or all(isinstance(h, logging.NullHandler) for h in LOGGER.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)
    else:
        LOGGER.setLevel(logging.CRITICAL + 1)


def set_verbose(enabled: bool = True) -> None:
    """Convenience toggle for both warnings and download verbosity at once."""
    set_warnings(enabled)
    set_download_verbose(enabled)


def warnings_enabled() -> bool:
    """Return whether inference warnings are currently enabled."""
    return _WARNINGS


def download_verbose_enabled() -> bool:
    """Return whether download verbosity is currently enabled."""
    return _DOWNLOAD_VERBOSE


def warn(msg: str) -> None:
    """Emit an FRBench inference warning when warnings are enabled."""
    if _WARNINGS:
        warnings.warn(f"[FRBench] {msg}", FRBenchWarning, stacklevel=2)


def info(msg: str) -> None:
    """Log a download status message when download verbosity is enabled."""
    if _DOWNLOAD_VERBOSE:
        LOGGER.info(msg)


class _NullBar:
    """No-op stand-in for a progress bar used when output is disabled."""

    def update(self, n: int = 0) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "_NullBar":
        return self

    def __exit__(self, *exc) -> None:
        pass


def progress_bar(total: Optional[int], desc: str):
    """Return a byte-oriented progress bar (context manager)."""
    if not _DOWNLOAD_VERBOSE:
        return _NullBar()
    from tqdm import tqdm

    return tqdm(total=total, desc=desc, unit="B", unit_scale=True, unit_divisor=1024, leave=True)


set_warnings(_WARNINGS)
set_download_verbose(_DOWNLOAD_VERBOSE)
