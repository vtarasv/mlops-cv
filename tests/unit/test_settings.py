"""Unit tests for the layered env config switch."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlops_cv.config.settings import (
    DEFAULT_ENV,
    Environment,
    current_env,
    env_files,
    load_settings,
)


def _write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


@pytest.fixture
def env_dir(tmp_path: Path) -> Path:
    """A base .env plus a .env.prod overlay with a conflicting value."""
    _write(tmp_path / ".env", "LOG_LEVEL=base")
    _write(tmp_path / ".env.prod", "LOG_LEVEL=prod")
    return tmp_path


def test_env_files_order(tmp_path: Path) -> None:
    assert env_files("dev", tmp_path) == (
        str(tmp_path / ".env"),
        str(tmp_path / ".env.dev"),
    )


def test_current_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENV", raising=False)
    assert current_env() == DEFAULT_ENV


def test_overlay_overrides_base(env_dir: Path) -> None:
    s = load_settings("prod", base_dir=env_dir)
    assert s.env == Environment.prod
    assert s.log_level == "prod"  # .env.prod wins over .env


def test_base_used_when_no_overlay(env_dir: Path) -> None:
    # No .env.local overlay exists -> fall back to the committed base value.
    assert load_settings("local", base_dir=env_dir).log_level == "base"


def test_os_env_overrides_files(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "from_os")
    assert load_settings("prod", base_dir=env_dir).log_level == "from_os"  # OS env beats both files


@pytest.mark.parametrize("env", ["local", "dev", "stage", "prod"])
def test_all_envs_load(env: str, env_dir: Path) -> None:
    assert load_settings(env, base_dir=env_dir).env == env


def test_model_default_when_no_files(tmp_path: Path) -> None:
    # No env files at all -> the model default applies.
    assert load_settings("local", base_dir=tmp_path).log_level == "INFO"
