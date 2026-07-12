"""Download and cache release assets described by ``manifest.json``."""
from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Sequence
import contextlib
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request

from .._config import get_cache, get_release, get_repo, is_detector_asset, parse_model_key
from .._exceptions import FRBenchAssetNotFoundError, FRBenchDownloadError
from .log import info, progress_bar


class ModelInfo(NamedTuple):
    """Structured description of one FR model in the manifest."""

    key: str
    backbone: str
    loss: str
    dataset: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_asset(name: str, *, refresh_manifest: bool = False) -> Dict[str, Any]:
    """Ensure asset *name* is cached and return its location + contents.

    Raises:
        FRBenchAssetNotFoundError: Key missing from manifest after refresh.
        FRBenchDownloadError: Download, extraction, or integrity check failed.
    """
    entry = _resolve_manifest_entry(name, refresh_manifest=refresh_manifest)
    if not _asset_cached(entry):
        info(f"Asset '{name}' not cached; downloading...")
        _fetch_asset(entry)
    return {
        "path": os.path.join(get_cache(), entry["target_name"]),
        "contents": entry["contents"],
    }


def list_assets(*, include_detectors: bool = True) -> List[str]:
    """Return sorted asset keys from the release manifest."""
    manifest = _asset_entries(_load_manifest())
    keys = sorted(manifest)
    if include_detectors:
        return keys
    return [k for k in keys if not is_detector_asset(k)]


def list_models() -> List[ModelInfo]:
    """Return structured FR model entries (detectors excluded)."""
    manifest = _asset_entries(_load_manifest())
    models: List[ModelInfo] = []
    for key in sorted(manifest):
        parsed = parse_model_key(key)
        if parsed is None:
            continue
        backbone, loss, dataset = parsed
        models.append(ModelInfo(key=key, backbone=backbone, loss=loss, dataset=dataset))
    return models


def download_assets(
    names: Optional[Sequence[str]] = None,
    *,
    download_all: bool = False,
    force: bool = False,
    refresh_manifest: bool = False,
) -> Dict[str, bool]:
    """Download one or more release assets into the cache directory."""
    if refresh_manifest:
        _download_manifest(force=True)

    manifest = _asset_entries(_load_manifest())
    if download_all:
        names = sorted(manifest)
    elif not names:
        raise FRBenchDownloadError("No assets specified. Pass asset names or download_all=True.")

    results: Dict[str, bool] = {}
    for name in names:
        try:
            if name not in manifest:
                raise FRBenchAssetNotFoundError(
                    f"Asset '{name}' not in manifest. Available: {sorted(manifest)}"
                )
            entry = manifest[name]
            if not force and _asset_cached(entry):
                info(f"Asset '{name}' already cached. Skipped.")
                results[name] = True
            else:
                if force:
                    _clear_cached(entry)
                info(f"Downloading '{name}'...")
                _fetch_asset(entry)
                results[name] = True
        except FRBenchDownloadError:
            results[name] = False
            raise
    return results


def refresh_manifest() -> Dict[str, Any]:
    """Re-download ``manifest.json`` from the release and return it."""
    return _download_manifest(force=True)


# ---------------------------------------------------------------------------
# Manifest and cache bookkeeping
# ---------------------------------------------------------------------------

def _manifest_path() -> str:
    return os.path.join(get_cache(), "manifest.json")


def _asset_entries(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Return manifest asset entries, excluding reserved metadata keys."""
    return {key: value for key, value in manifest.items() if not key.startswith("__")}


def _download_manifest(*, force: bool = False) -> Dict[str, Any]:
    path = _manifest_path()
    if force or not os.path.exists(path):
        info("Downloading manifest.json...")
        os.makedirs(get_cache(), exist_ok=True)
        _download_release_file("manifest.json", path)
        _verify_file_sha(path, None)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_manifest(*, force: bool = False) -> Dict[str, Any]:
    path = _manifest_path()
    if force or not os.path.exists(path):
        return _download_manifest(force=True)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_manifest_entry(name: str, *, refresh_manifest: bool = False) -> Dict[str, Any]:
    manifest = _asset_entries(_load_manifest(force=refresh_manifest))
    if name in manifest:
        return manifest[name]
    # One automatic refresh attempt when key is missing.
    manifest = _asset_entries(_download_manifest(force=True))
    if name not in manifest:
        raise FRBenchAssetNotFoundError(
            f"Asset '{name}' not in manifest. Available: {sorted(manifest)}"
        )
    return manifest[name]


def _asset_cached(entry: Dict[str, Any]) -> bool:
    folder = os.path.join(get_cache(), entry["target_name"])
    if not os.path.isdir(folder):
        return False
    return all(
        os.path.isfile(os.path.join(folder, rel))
        for rel in entry.get("contents", {}).values()
    )


def _clear_cached(entry: Dict[str, Any]) -> None:
    folder = os.path.join(get_cache(), entry["target_name"])
    if os.path.isdir(folder):
        shutil.rmtree(folder)


def _lock_path(entry: Dict[str, Any]) -> str:
    return os.path.join(get_cache(), f".{entry['target_name']}.lock")


@contextlib.contextmanager
def _asset_lock(entry: Dict[str, Any]):
    os.makedirs(get_cache(), exist_ok=True)
    lock = _lock_path(entry)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR)
    try:
        if hasattr(os, "lockf"):
            os.lockf(fd, os.F_LOCK, 0)
        yield
    finally:
        if hasattr(os, "lockf"):
            os.lockf(fd, os.F_ULOCK, 0)
        os.close(fd)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _verify_file_sha(path: str, expected: Optional[str]) -> None:
    if not expected:
        return
    actual = _sha256_file(path)
    if actual != expected:
        if os.path.exists(path):
            os.remove(path)
        raise FRBenchDownloadError(
            f"SHA-256 mismatch for {os.path.basename(path)}: expected {expected}, got {actual}"
        )


def _fetch_asset(entry: Dict[str, Any]) -> None:
    with _asset_lock(entry):
        if _asset_cached(entry):
            return
        asset = entry["asset_name"]
        archive = os.path.join(get_cache(), asset)
        os.makedirs(get_cache(), exist_ok=True)
        _download_release_file(asset, archive, entry.get("size"))
        _verify_file_sha(archive, entry.get("sha"))
        try:
            with tempfile.TemporaryDirectory(dir=get_cache()) as tmpdir:
                with tarfile.open(archive, "r:gz") as tar:
                    top = os.path.commonpath(tar.getnames())
                    tar.extractall(tmpdir, filter="data")
                extracted = os.path.join(tmpdir, top)
                target = os.path.join(get_cache(), entry["target_name"])
                if os.path.exists(target):
                    backup = target + time.strftime(".backup_%Y%m%d%H%M%S")
                    os.rename(target, backup)
                    _cleanup_backups(target)
                os.replace(extracted, target)
        except (tarfile.TarError, OSError) as e:
            raise FRBenchDownloadError(f"Failed to extract {asset}: {e}") from e
        finally:
            if os.path.exists(archive):
                os.remove(archive)
        if not _asset_cached(entry):
            raise FRBenchDownloadError(f"Asset '{entry['target_name']}' incomplete after extraction.")


def _cleanup_backups(target: str) -> None:
    parent = os.path.dirname(target) or "."
    base = os.path.basename(target)
    backups = sorted(
        f for f in os.listdir(parent) if f.startswith(base + ".backup_")
    )
    for old in backups[:-1]:
        path = os.path.join(parent, old)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# HTTP download with progress, retries, and backoff
# ---------------------------------------------------------------------------

def _http_download(
    url: str,
    dest: str,
    desc: str,
    total: Optional[int] = None,
    headers: Optional[Dict[str, str]] = None,
    *,
    max_retries: int = 3,
) -> None:
    req_headers = {"User-Agent": "frbench", **(headers or {})}
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
                if total is None and resp.headers.get("Content-Length"):
                    total = int(resp.headers["Content-Length"])
                with progress_bar(total, desc) as bar:
                    for chunk in iter(lambda: resp.read(1 << 20), b""):
                        out.write(chunk)
                        bar.update(len(chunk))
            if os.path.getsize(dest) == 0:
                raise FRBenchDownloadError(f"Downloaded file is empty: {desc}")
            return
        except Exception as e:
            last_err = e
            if os.path.exists(dest):
                os.remove(dest)
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                info(f"Download failed ({desc}), retrying in {delay}s: {e}")
                time.sleep(delay)
    raise FRBenchDownloadError(f"Download failed ({url}): {last_err}") from last_err


def _download_release_file(asset: str, dest: str, size: Optional[int] = None) -> None:
    """Download an asset from the configured public GitHub release."""
    url = f"https://github.com/{get_repo()}/releases/download/{get_release()}/{asset}"
    _http_download(url, dest, desc=asset, total=size)
