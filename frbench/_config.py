"""Runtime configuration for FRBench.

Every setting is resolved through :func:`get_setting` with the following
precedence (lowest to highest):

1. Built-in defaults.
2. The persisted config file (``~/.frbench/config.json`` by default; its
   location can be overridden with the ``FRBENCH_CONFIG`` environment variable
   and is independent of the ``cache`` setting).
3. Environment variables (``FRBENCH_CACHE``, ``FRBENCH_REPO``, ...).
4. Runtime values set through :func:`configure` (or ``frbench.set_verbose``
   and friends).
5. Active :func:`configure_scoped` overrides (innermost wins).

Use :func:`configure` with ``persist=True`` to write settings to the config
file, and :func:`configure_scoped` as a context manager for temporary
overrides.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import threading
from typing import Any, Dict, Iterator, Optional, Tuple

from ._exceptions import FRBenchConfigError

__version__ = "1.1.2"

#: Sentinel distinguishing "argument not given" from an explicit ``None``.
_UNSET = object()

_DEFAULT_CONFIG_FILE = os.path.join("~", ".frbench", "config.json")

_DEFAULTS: Dict[str, Any] = {
    "cache": os.path.join("~", ".frbench"),
    "repo": "HKU-TASR/FRBench",
    "release": "weights-v1.0.0",
    "warnings": True,
    "download_verbose": True,
    "update_check": True,
}

_BOOL_KEYS = frozenset({"warnings", "download_verbose", "update_check"})

_ENV_VARS: Dict[str, str] = {
    "cache": "FRBENCH_CACHE",
    "repo": "FRBENCH_REPO",
    "release": "FRBENCH_RELEASE",
    "warnings": "FRBENCH_WARNINGS",
    "download_verbose": "FRBENCH_DOWNLOAD_VERBOSE",
    "update_check": "FRBENCH_UPDATE_CHECK",
}

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

_DETECTOR_PREFIX = "retinaface_"

_LOCK = threading.Lock()
_RUNTIME: Dict[str, Any] = {}
_SCOPED: contextvars.ContextVar[Tuple[Dict[str, Any], ...]] = contextvars.ContextVar(
    "frbench_scoped_config", default=()
)
# Cached parsed config file: (path, mtime, data).
_FILE_CACHE: Optional[Tuple[str, float, Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Value parsing / validation
# ---------------------------------------------------------------------------

def _parse_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUTHY:
            return True
        if lowered in _FALSY:
            return False
    raise FRBenchConfigError(
        f"Invalid boolean for setting '{key}': {value!r}. "
        f"Use one of {sorted(_TRUTHY)} / {sorted(_FALSY)}."
    )


def _coerce(key: str, value: Any) -> Any:
    if key not in _DEFAULTS:
        raise FRBenchConfigError(
            f"Unknown FRBench setting '{key}'. Valid settings: {sorted(_DEFAULTS)}"
        )
    if key in _BOOL_KEYS:
        return _parse_bool(key, value)
    if not isinstance(value, str):
        raise FRBenchConfigError(
            f"Setting '{key}' expects a string, got {type(value).__name__}: {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Layer readers
# ---------------------------------------------------------------------------

def config_file_path() -> str:
    """Return the config file location (``FRBENCH_CONFIG`` env var or default)."""
    return os.path.expanduser(
        os.path.expandvars(os.environ.get("FRBENCH_CONFIG", _DEFAULT_CONFIG_FILE))
    )


def _read_config_file() -> Dict[str, Any]:
    """Read (and cache by mtime) the persisted config file. Missing file = {}."""
    global _FILE_CACHE
    path = config_file_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    with _LOCK:
        if _FILE_CACHE is not None and _FILE_CACHE[0] == path and _FILE_CACHE[1] == mtime:
            return _FILE_CACHE[2]
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        raise FRBenchConfigError(f"Could not read FRBench config file {path}: {e}") from e
    if not isinstance(raw, dict):
        raise FRBenchConfigError(f"FRBench config file {path} must contain a JSON object.")
    data = {k: _coerce(k, v) for k, v in raw.items() if k in _DEFAULTS}
    with _LOCK:
        _FILE_CACHE = (path, mtime, data)
    return data


def _write_config_file(updates: Dict[str, Any], removals: Tuple[str, ...] = ()) -> None:
    """Merge *updates* into the config file (creating it if needed)."""
    global _FILE_CACHE
    path = config_file_path()
    try:
        current = dict(_read_config_file())
    except FRBenchConfigError:
        current = {}
    current.update(updates)
    for key in removals:
        current.pop(key, None)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    with _LOCK:
        _FILE_CACHE = None


def _env_value(key: str) -> Optional[Any]:
    raw = os.environ.get(_ENV_VARS[key])
    if key == "update_check" and raw is None:
        # Legacy negative switch, kept for backward compatibility.
        legacy = os.environ.get("FRBENCH_NO_UPDATE_CHECK")
        if legacy is not None and legacy.strip().lower() in _TRUTHY:
            return False
        return None
    if raw is None:
        return None
    return _coerce(key, raw)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def get_setting(key: str) -> Any:
    """Resolve setting *key* through all configuration layers."""
    if key not in _DEFAULTS:
        raise FRBenchConfigError(
            f"Unknown FRBench setting '{key}'. Valid settings: {sorted(_DEFAULTS)}"
        )
    for layer in reversed(_SCOPED.get()):
        if key in layer:
            return layer[key]
    if key in _RUNTIME:
        return _RUNTIME[key]
    env = _env_value(key)
    if env is not None:
        return env
    file_cfg = _read_config_file()
    if key in file_cfg:
        return file_cfg[key]
    return _DEFAULTS[key]


def describe_settings() -> Dict[str, Dict[str, Any]]:
    """Return ``{key: {"value": ..., "source": ...}}`` for every setting."""
    scoped_layers = _SCOPED.get()
    file_cfg = _read_config_file()
    result: Dict[str, Dict[str, Any]] = {}
    for key in _DEFAULTS:
        source = "default"
        if key in file_cfg:
            source = "config file"
        if _env_value(key) is not None:
            source = "environment"
        if key in _RUNTIME:
            source = "runtime"
        if any(key in layer for layer in scoped_layers):
            source = "scoped override"
        result[key] = {"value": get_setting(key), "source": source}
    return result


# ---------------------------------------------------------------------------
# Public configuration API
# ---------------------------------------------------------------------------

def _collect(
    cache: Any,
    repo: Any,
    release: Any,
    warnings: Any,
    download_verbose: Any,
    update_check: Any,
    verbose: Any,
) -> Dict[str, Any]:
    """Expand ``verbose`` and gather explicitly provided settings (may be None)."""
    provided: Dict[str, Any] = {}
    if verbose is not _UNSET:
        provided["warnings"] = verbose
        provided["download_verbose"] = verbose
    for key, value in (
        ("cache", cache),
        ("repo", repo),
        ("release", release),
        ("warnings", warnings),
        ("download_verbose", download_verbose),
        ("update_check", update_check),
    ):
        if value is not _UNSET:
            provided[key] = value
    return {k: (None if v is None else _coerce(k, v)) for k, v in provided.items()}


def configure(
    *,
    cache: Any = _UNSET,
    repo: Any = _UNSET,
    release: Any = _UNSET,
    warnings: Any = _UNSET,
    download_verbose: Any = _UNSET,
    update_check: Any = _UNSET,
    verbose: Any = _UNSET,
    persist: bool = False,
) -> None:
    """Set FRBench settings for this process (optionally persisted).

    Args:
        cache: Directory for downloaded weights (``~`` and env vars expanded).
        repo: GitHub repo slug the assets are downloaded from.
        release: GitHub release tag for the weight assets.
        warnings: Enable/disable inference warnings.
        download_verbose: Enable/disable download logs and progress bars.
        update_check: Enable/disable the once-a-day update reminder.
        verbose: Convenience switch setting both ``warnings`` and
            ``download_verbose`` (explicit values for those win).
        persist: If ``True``, also write the given settings to the config file
            (``frbench._config.config_file_path()``) so they survive across
            processes. Passing ``None`` for a setting resets it: the runtime
            value is dropped and, with ``persist=True``, the persisted value
            is removed as well.

    Examples:
        >>> frbench.configure(verbose=False)                      # this process
        >>> frbench.configure(cache="/data/frbench", persist=True)  # permanent
        >>> frbench.configure(cache=None, persist=True)           # back to default
    """
    provided = _collect(cache, repo, release, warnings, download_verbose, update_check, verbose)
    if not provided:
        return
    with _LOCK:
        for key, value in provided.items():
            if value is None:
                _RUNTIME.pop(key, None)
            else:
                _RUNTIME[key] = value
    if persist:
        updates = {k: v for k, v in provided.items() if v is not None}
        removals = tuple(k for k, v in provided.items() if v is None)
        _write_config_file(updates, removals)


@contextlib.contextmanager
def configure_scoped(
    *,
    cache: Any = _UNSET,
    repo: Any = _UNSET,
    release: Any = _UNSET,
    warnings: Any = _UNSET,
    download_verbose: Any = _UNSET,
    update_check: Any = _UNSET,
    verbose: Any = _UNSET,
) -> Iterator[None]:
    """Temporarily override settings within a ``with`` block (thread-safe).

    Accepts the same settings as :func:`configure`. Overrides apply only inside
    the block (and only in the current thread/async task) and are undone on
    exit, no matter how the block exits.

    Example:
        >>> with frbench.configure_scoped(verbose=False, cache="/tmp/frbench"):
        ...     fr = frbench.FR("mobilefacenet", "arcface", "ms1m")
    """
    provided = _collect(cache, repo, release, warnings, download_verbose, update_check, verbose)
    overrides = {k: v for k, v in provided.items() if v is not None}
    token = _SCOPED.set(_SCOPED.get() + (overrides,))
    try:
        yield
    finally:
        _SCOPED.reset(token)


# ---------------------------------------------------------------------------
# Convenience getters used throughout the package
# ---------------------------------------------------------------------------

def get_cache() -> str:
    """Return the active cache directory (``~`` and env vars expanded)."""
    return os.path.expanduser(os.path.expandvars(get_setting("cache")))


def get_repo() -> str:
    """Return the active GitHub repo slug."""
    return get_setting("repo")


def get_release() -> str:
    """Return the active GitHub release tag."""
    return get_setting("release")


def __getattr__(name: str) -> Any:
    """Expose ``CACHE`` / ``REPO`` / ``RELEASE`` as live, read-only views."""
    if name == "CACHE":
        return get_cache()
    if name == "REPO":
        return get_repo()
    if name == "RELEASE":
        return get_release()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
