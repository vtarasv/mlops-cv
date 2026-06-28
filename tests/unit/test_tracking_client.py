"""Unit tests for MLflow client-config resolution (no server, no mlflow dep)."""

from __future__ import annotations

import pytest

from mlops_cv.config import load_settings
from mlops_cv.tracking import client


def test_defaults_from_settings() -> None:
    s = load_settings("local", base_dir="/nonexistent")  # no env files -> model defaults
    assert client.tracking_uri(s) == "http://localhost:5000"


def test_mlflow_env_only_tracking_uri() -> None:
    s = load_settings("local", base_dir="/nonexistent")
    assert client.mlflow_env(s) == {"MLFLOW_TRACKING_URI": "http://localhost:5000"}


def test_nested_env_override_flows_through(monkeypatch: pytest.MonkeyPatch) -> None:
    # MLFLOW__TRACKING_URI -> settings.mlflow.tracking_uri (env_nested_delimiter="__").
    monkeypatch.setenv("MLFLOW__TRACKING_URI", "http://mlflow.internal:5000")
    s = load_settings("local", base_dir="/nonexistent")
    assert client.tracking_uri(s) == "http://mlflow.internal:5000"
    assert client.mlflow_env(s)["MLFLOW_TRACKING_URI"] == "http://mlflow.internal:5000"


def test_configure_exports_to_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW__TRACKING_URI", "http://host:1234")
    s = load_settings("local", base_dir="/nonexistent")
    client.configure(s)
    import os

    assert os.environ["MLFLOW_TRACKING_URI"] == "http://host:1234"
