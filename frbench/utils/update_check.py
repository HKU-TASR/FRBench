"""Lightweight, best-effort checks for newer FRBench code and weights."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import json
import os
import re
import time
import urllib.request

from .._config import __version__, configure, get_cache, get_release, get_repo, get_setting
from .log import warn

_CACHE_TTL_SECONDS = 24 * 60 * 60
_PYPI_URL = "https://pypi.org/pypi/frbench/json"
_SEMVER_RE = re.compile(r"^(?:weights-v|v)?(\d+)\.(\d+)\.(\d+)$")

_CHECKED_THIS_PROCESS = False


def set_update_check(enabled: bool = True, *, persist: bool = False) -> None:
    """Enable or disable automatic update reminders.

    Args:
        enabled: New value for the switch.
        persist: If ``True``, also write the value to the FRBench config file
            so it applies to future processes.
    """
    configure(update_check=bool(enabled), persist=persist)


def update_check_enabled() -> bool:
    """Return whether automatic update reminders are enabled."""
    return bool(get_setting("update_check"))


def _version_key(version: str) -> Optional[Tuple[int, int, int]]:
    match = _SEMVER_RE.fullmatch(version)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def _is_newer(candidate: Optional[str], current: str) -> bool:
    if candidate is None:
        return False
    candidate_key = _version_key(candidate)
    current_key = _version_key(current)
    return candidate_key is not None and current_key is not None and candidate_key > current_key


def _cache_path() -> str:
    return os.path.join(get_cache(), "update_check.json")


def _read_cache() -> Optional[Dict[str, Any]]:
    try:
        with open(_cache_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        checked_at = float(data["checked_at"])
        if time.time() - checked_at <= _CACHE_TTL_SECONDS:
            return data
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _write_cache(data: Dict[str, Any]) -> None:
    try:
        os.makedirs(get_cache(), exist_ok=True)
        path = _cache_path()
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        os.replace(temporary, path)
    except OSError:
        pass


def _get_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json, application/json",
            "User-Agent": f"frbench/{__version__}",
        },
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        return json.load(response)


def _fetch_latest() -> Dict[str, Any]:
    latest_code: Optional[str] = None
    latest_weights: Optional[str] = None

    try:
        payload = _get_json(_PYPI_URL)
        latest_code = str(payload["info"]["version"])
    except (OSError, ValueError, KeyError, TypeError):
        pass

    try:
        releases = _get_json(f"https://api.github.com/repos/{get_repo()}/releases?per_page=100")
        weight_tags = [
            str(release["tag_name"])
            for release in releases
            if not release.get("draft") and _version_key(str(release.get("tag_name", ""))) is not None
            and str(release.get("tag_name", "")).startswith("weights-v")
        ]
        latest_weights = max(weight_tags, key=lambda tag: _version_key(tag) or (0, 0, 0), default=None)
    except (OSError, ValueError, KeyError, TypeError):
        pass

    result = {
        "checked_at": time.time(),
        "latest_code": latest_code,
        "latest_weights": latest_weights,
    }
    _write_cache(result)
    return result


def check_for_updates(*, force: bool = False) -> None:
    """Warn about newer package or weights releases; never raise on failure."""
    global _CHECKED_THIS_PROCESS
    if not update_check_enabled() or (_CHECKED_THIS_PROCESS and not force):
        return
    _CHECKED_THIS_PROCESS = True

    try:
        result = None if force else _read_cache()
        if result is None:
            result = _fetch_latest()

        latest_code = result.get("latest_code")
        if _is_newer(latest_code, __version__):
            warn(
                f"frbench {latest_code} is available (you have {__version__}). "
                "Upgrade with: pip install -U frbench"
            )

        latest_weights = result.get("latest_weights")
        if _is_newer(latest_weights, get_release()):
            warn(
                f"newer weights release {latest_weights} is available "
                f"(you use {get_release()}). Upgrade frbench for a compatible default."
            )
    except Exception:
        # Update reminders must never prevent normal FRBench use.
        return
