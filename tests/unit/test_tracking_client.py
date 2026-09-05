"""Unit tests for the MLflow session module: config resolution and the connect choreography
(no server; mlflow faked where needed)."""

from __future__ import annotations

import sys
import types

import pytest

from mlops_cv.config import load_settings
from mlops_cv.tracking import client


def test_configure_exports_only_the_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proxied artifacts: no S3 endpoint or AWS credentials are needed client-side."""
    import os

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    client.configure(load_settings(base_dir="/nonexistent"))  # no env files -> model defaults
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://localhost:5000"

    # MLFLOW__TRACKING_URI -> settings.mlflow.tracking_uri (env_nested_delimiter="__").
    monkeypatch.setenv("MLFLOW__TRACKING_URI", "http://host:1234")
    client.configure(load_settings(base_dir="/nonexistent"))
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


def test_registry_hands_back_an_injected_client_untouched() -> None:
    sentinel = object()
    assert client.registry(load_settings(base_dir="/nonexistent"), injected=sentinel) is sentinel


def test_registry_configures_the_session_before_building_the_real_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordering invariant of every registry walk, owned here: URI exported, then the client."""
    import os

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW__TRACKING_URI", "http://host:1234")
    seen: dict[str, str | None] = {}

    class FakeClient:
        def __init__(self) -> None:
            seen["env"] = os.environ.get("MLFLOW_TRACKING_URI")

    fake = types.ModuleType("mlflow")
    fake.MlflowClient = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    assert isinstance(client.registry(load_settings(base_dir="/nonexistent")), FakeClient)
    assert seen["env"] == "http://host:1234"
