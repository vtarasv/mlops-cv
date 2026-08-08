"""Unit tests for the MLflow session module: config resolution and the connect choreography
(no server; mlflow faked where needed)."""

from __future__ import annotations

import sys
import types

import pytest

from mlops_cv.config import load_settings
from mlops_cv.tracking import client


def test_defaults_from_settings() -> None:
    s = load_settings(base_dir="/nonexistent")  # no env files -> model defaults
    assert client.tracking_uri(s) == "http://localhost:5000"


def test_mlflow_env_only_tracking_uri() -> None:
    s = load_settings(base_dir="/nonexistent")
    assert client.mlflow_env(s) == {"MLFLOW_TRACKING_URI": "http://localhost:5000"}


def test_nested_env_override_flows_through(monkeypatch: pytest.MonkeyPatch) -> None:
    # MLFLOW__TRACKING_URI -> settings.mlflow.tracking_uri (env_nested_delimiter="__").
    monkeypatch.setenv("MLFLOW__TRACKING_URI", "http://mlflow.internal:5000")
    s = load_settings(base_dir="/nonexistent")
    assert client.tracking_uri(s) == "http://mlflow.internal:5000"
    assert client.mlflow_env(s)["MLFLOW_TRACKING_URI"] == "http://mlflow.internal:5000"


def test_configure_exports_to_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW__TRACKING_URI", "http://host:1234")
    s = load_settings(base_dir="/nonexistent")
    client.configure(s)
    import os

    assert os.environ["MLFLOW_TRACKING_URI"] == "http://host:1234"


def test_champion_uri_names_the_registered_model_at_its_alias() -> None:
    s = load_settings(base_dir="/nonexistent")
    assert client.champion_uri(s) == "models:/aerial-object-detector@champion"


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A fake ``mlflow`` module recording what connect sets on it."""
    fake = types.ModuleType("mlflow")
    fake.set_tracking_uri = lambda uri: setattr(fake, "uri", uri)  # type: ignore[attr-defined]
    fake.set_experiment = lambda name: setattr(fake, "experiment", name)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return fake


def test_connect_exports_env_and_configures_the_module(
    monkeypatch: pytest.MonkeyPatch, fake_mlflow: types.ModuleType
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW__TRACKING_URI", "http://host:1234")
    s = load_settings(base_dir="/nonexistent")
    mlflow = client.connect(s)
    import os

    assert mlflow is fake_mlflow
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://host:1234"  # exported before import
    assert mlflow.uri == "http://host:1234"  # type: ignore[attr-defined]
    assert mlflow.experiment == s.mlflow.experiment  # type: ignore[attr-defined]


def test_connect_without_experiment_joins_existing_runs_only(
    fake_mlflow: types.ModuleType,
) -> None:
    """The on-device harness only joins the desktop's record run — no experiment selection."""
    mlflow = client.connect(load_settings(base_dir="/nonexistent"), experiment=False)
    assert mlflow.uri == "http://localhost:5000"  # type: ignore[attr-defined]
    assert not hasattr(mlflow, "experiment")
