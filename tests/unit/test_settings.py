"""Unit tests for settings: single .env + OS-env precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlops_cv.config.settings import load_settings


def _write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate from the caller's shell so LOG_LEVEL comes only from each test's inputs."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)


def test_dotenv_values_loaded(tmp_path: Path) -> None:
    _write(tmp_path / ".env", "LOG_LEVEL=DEBUG")
    assert load_settings(base_dir=tmp_path).log_level == "DEBUG"


def test_os_env_overrides_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path / ".env", "LOG_LEVEL=from_file")
    monkeypatch.setenv("LOG_LEVEL", "from_os")
    assert load_settings(base_dir=tmp_path).log_level == "from_os"  # OS env beats .env


def test_nested_data_override_from_os(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA__SUBSET_DIR", "/somewhere/outside/the/repo")
    assert load_settings(base_dir=tmp_path).data.subset_dir == Path("/somewhere/outside/the/repo")


def test_model_defaults_when_no_file(tmp_path: Path) -> None:
    assert load_settings(base_dir=tmp_path).log_level == "INFO"  # empty dir -> model defaults


def test_mlflow_champion_alias_default(tmp_path: Path) -> None:
    # The registry alias marking the deployed model for the champion/challenger gate.
    assert load_settings(base_dir=tmp_path).mlflow.champion_alias == "champion"
