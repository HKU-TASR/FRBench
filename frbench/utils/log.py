"""Verbosity control, diagnostics, and download progress.

This module is the single source of truth for the package's user-facing output.
There are two independent switches (both enabled by default):

- Inference warnings (e.g. no face detected, unsupported TTA) go through the
  stdlib :mod:`warnings` module under the :class:`FRBenchWarning` category.
- Download progress/status (log messages + the ``tqdm`` bar) is a separate
  channel toggled with :func:`set_download_verbose`.

Both switches are backed by the FRBench configuration system: they can be set
via environment variables (``FRBENCH_WARNINGS`` / ``FRBENCH_DOWNLOAD_VERBOSE``),
persisted with ``persist=True``, or temporarily overridden with
:func:`frbench.configure_scoped`.
"""
from __future__ import annotations

import logging
from typing import Optional
import warnings

from .._config import configure, get_setting

LOGGER = logging.getLogger("frbench")
if not any(not isinstance(h, logging.NullHandler) for h in LOGGER.handlers):
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


class FRBenchWarning(UserWarning):
    """Category for FRBench inference-time warnings (filterable via ``warnings``)."""


# One-time, category-scoped default so repeated warnings (e.g. "no detection"
# in a long loop) stay visible. User filters registered afterwards take
# precedence; the switches below never touch the global filter list again.
warnings.filterwarnings("always", category=FRBenchWarning)


def set_warnings(enabled: bool = True, *, persist: bool = False) -> None:
    """Enable or disable FRBench inference warnings.

    Args:
        enabled: New value for the switch.
        persist: If ``True``, also write the value to the FRBench config file
            so it applies to future processes.
    """
    configure(warnings=bool(enabled), persist=persist)


def set_download_verbose(enabled: bool = True, *, persist: bool = False) -> None:
    """Enable or disable download status messages and progress bars.

    Args:
        enabled: New value for the switch.
        persist: If ``True``, also write the value to the FRBench config file
            so it applies to future processes.
    """
    configure(download_verbose=bool(enabled), persist=persist)


def set_verbose(enabled: bool = True, *, persist: bool = False) -> None:
    """Convenience toggle for both warnings and download verbosity at once."""
    configure(verbose=bool(enabled), persist=persist)


def warnings_enabled() -> bool:
    """Return whether inference warnings are currently enabled."""
    return bool(get_setting("warnings"))


def download_verbose_enabled() -> bool:
    """Return whether download verbosity is currently enabled."""
    return bool(get_setting("download_verbose"))


def warn(msg: str) -> None:
    """Emit an FRBench inference warning when warnings are enabled."""
    if warnings_enabled():
        warnings.warn(f"[FRBench] {msg}", FRBenchWarning, stacklevel=2)


def info(msg: str) -> None:
    """Log a download status message when download verbosity is enabled."""
    if download_verbose_enabled():
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
    if not download_verbose_enabled():
        return _NullBar()
    from tqdm import tqdm

    return tqdm(total=total, desc=desc, unit="B", unit_scale=True, unit_divisor=1024, leave=True)
