"""Configuration precedence, persistence, and scoped overrides (offline)."""
import json
import os

import pytest

import frbench
from frbench import _config
from frbench._exceptions import FRBenchConfigError


def test_defaults(monkeypatch):
    monkeypatch.delenv("FRBENCH_CACHE", raising=False)
    assert frbench.CACHE == os.path.expanduser(os.path.join("~", ".frbench"))
    assert frbench.REPO == "HKU-TASR/FRBench"
    assert _config.get_setting("warnings") is True
    assert _config.get_setting("download_verbose") is True
    assert _config.get_setting("update_check") is False  # disabled by conftest env


def test_env_overrides_file(monkeypatch, tmp_path):
    _config._write_config_file({"repo": "file/repo"})
    assert frbench.REPO == "file/repo"
    monkeypatch.setenv("FRBENCH_REPO", "env/repo")
    assert frbench.REPO == "env/repo"


def test_runtime_overrides_env(monkeypatch):
    monkeypatch.setenv("FRBENCH_REPO", "env/repo")
    frbench.configure(repo="runtime/repo")
    assert frbench.REPO == "runtime/repo"
    frbench.configure(repo=None)  # reset drops the runtime layer
    assert frbench.REPO == "env/repo"


def test_scoped_overrides_everything_and_restores(monkeypatch):
    frbench.configure(repo="runtime/repo")
    with frbench.configure_scoped(repo="scoped/repo"):
        assert frbench.REPO == "scoped/repo"
        with frbench.configure_scoped(repo="inner/repo"):
            assert frbench.REPO == "inner/repo"
        assert frbench.REPO == "scoped/repo"
    assert frbench.REPO == "runtime/repo"


def test_scoped_restores_on_exception():
    try:
        with frbench.configure_scoped(cache="/tmp/scoped-cache"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert frbench.CACHE != "/tmp/scoped-cache"


def test_persist_roundtrip(tmp_path, monkeypatch):
    monkeypatch.delenv("FRBENCH_CACHE", raising=False)  # env outranks the file
    frbench.configure(cache="/data/frbench", download_verbose=False, persist=True)
    path = _config.config_file_path()
    with open(path) as f:
        stored = json.load(f)
    assert stored == {"cache": "/data/frbench", "download_verbose": False}

    # A "fresh process": clear the runtime layer, the file still applies.
    with _config._LOCK:
        _config._RUNTIME.clear()
    assert frbench.CACHE == "/data/frbench"
    assert frbench.download_verbose_enabled() is False

    # Unset removes the persisted key.
    frbench.configure(cache=None, persist=True)
    with open(path) as f:
        assert "cache" not in json.load(f)


def test_verbose_expands_to_both_switches():
    frbench.configure(verbose=False)
    assert not frbench.warnings_enabled()
    assert not frbench.download_verbose_enabled()
    frbench.configure(verbose=True, download_verbose=False)  # explicit key wins
    assert frbench.warnings_enabled()
    assert not frbench.download_verbose_enabled()


def test_set_helpers_are_config_backed():
    frbench.set_warnings(False)
    assert _config.get_setting("warnings") is False
    frbench.set_download_verbose(False)
    assert _config.get_setting("download_verbose") is False
    frbench.set_verbose(True)
    assert frbench.warnings_enabled() and frbench.download_verbose_enabled()
    frbench.set_update_check(True)
    assert frbench.update_check_enabled() is True


def test_bool_env_parsing(monkeypatch):
    monkeypatch.setenv("FRBENCH_WARNINGS", "off")
    assert frbench.warnings_enabled() is False
    monkeypatch.setenv("FRBENCH_WARNINGS", "ON")
    assert frbench.warnings_enabled() is True
    monkeypatch.setenv("FRBENCH_WARNINGS", "banana")
    with pytest.raises(FRBenchConfigError):
        frbench.warnings_enabled()


def test_legacy_no_update_check_env(monkeypatch):
    monkeypatch.delenv("FRBENCH_UPDATE_CHECK", raising=False)
    monkeypatch.setenv("FRBENCH_NO_UPDATE_CHECK", "1")
    assert frbench.update_check_enabled() is False


def test_unknown_setting_rejected():
    with pytest.raises(TypeError):
        frbench.configure(bogus_key=1)
    with pytest.raises(FRBenchConfigError):
        _config.get_setting("bogus_key")


def test_module_constants_are_live_and_readonly_views():
    before = frbench.CACHE
    frbench.configure(cache="/tmp/live-view")
    assert frbench.CACHE == "/tmp/live-view"
    frbench.configure(cache=None)
    assert frbench.CACHE == before


def test_describe_settings_sources(monkeypatch):
    monkeypatch.setenv("FRBENCH_REPO", "env/repo")
    frbench.configure(release="weights-v9.9.9")
    described = _config.describe_settings()
    assert described["repo"]["source"] == "environment"
    assert described["release"]["source"] == "runtime"
    assert described["warnings"]["source"] == "default"
