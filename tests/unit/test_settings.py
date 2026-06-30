"""Unit tests for settings: single .env + OS-env precedence; ENV identifier field."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlops_cv.config.settings import Environment, load_settings


def _write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate from the caller's shell so ENV/LOG_LEVEL come only from each test's inputs."""
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)


def test_dotenv_values_loaded(tmp_path: Path) -> None:
    _write(tmp_path / ".env", "ENV=prod\nLOG_LEVEL=DEBUG")
    s = load_settings(base_dir=tmp_path)
    assert s.env == Environment.prod
    assert s.log_level == "DEBUG"


def test_os_env_overrides_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path / ".env", "LOG_LEVEL=from_file")
    monkeypatch.setenv("LOG_LEVEL", "from_os")
    assert load_settings(base_dir=tmp_path).log_level == "from_os"  # OS env beats .env


def test_env_field_from_os(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # `env` is a plain field populated from the ENV variable (no file-selection role).
    monkeypatch.setenv("ENV", "stage")
    assert load_settings(base_dir=tmp_path).env == Environment.stage


def test_model_defaults_when_no_file(tmp_path: Path) -> None:
    s = load_settings(base_dir=tmp_path)  # empty dir -> model defaults
    assert s.log_level == "INFO"
    assert s.env == Environment.local


@pytest.mark.parametrize("env", ["local", "dev", "stage", "prod"])
def test_all_env_values_valid(env: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", env)
    assert load_settings(base_dir=tmp_path).env == env
