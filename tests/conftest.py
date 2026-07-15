"""Shared fixtures: isolate every test from the user's real FRBench state."""
import pytest

import frbench
from frbench import _config


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point the config file and cache at a temp dir; clear runtime overrides."""
    monkeypatch.setenv("FRBENCH_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("FRBENCH_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FRBENCH_UPDATE_CHECK", "0")
    for var in ("FRBENCH_REPO", "FRBENCH_RELEASE", "FRBENCH_WARNINGS",
                "FRBENCH_DOWNLOAD_VERBOSE", "FRBENCH_NO_UPDATE_CHECK"):
        monkeypatch.delenv(var, raising=False)
    with _config._LOCK:
        saved_runtime = dict(_config._RUNTIME)
        _config._RUNTIME.clear()
        _config._FILE_CACHE = None
    yield
    with _config._LOCK:
        _config._RUNTIME.clear()
        _config._RUNTIME.update(saved_runtime)
        _config._FILE_CACHE = None
