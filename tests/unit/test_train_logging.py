"""Unit tests for ``train._register_model`` wiring, mlflow faked (CI-safe)."""

from __future__ import annotations

import sys
import types

import pytest

from mlops_cv.training.train import _register_model


class _FakeVersion:
    def __init__(self, version: str) -> None:
        self.version = version


@pytest.fixture
def registry_calls(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Fake ``mlflow`` + ``mlflow.exceptions``; the ``MlflowClient`` records registry calls."""
    recorded: dict = {"models_created": 0, "versions_created": 0}

    class FakeRestException(Exception):
        pass

    class FakeClient:
        def create_registered_model(self, name: str) -> None:
            recorded["models_created"] += 1
            if recorded["models_created"] > 1:  # the real registry rejects duplicates
                raise FakeRestException("RESOURCE_ALREADY_EXISTS")

        def create_model_version(self, name: str, source: str, run_id: str) -> _FakeVersion:
            recorded["versions_created"] += 1
            recorded["version_args"] = (name, source, run_id)
            return _FakeVersion(str(recorded["versions_created"]))

    fake = types.ModuleType("mlflow")
    fake.MlflowClient = FakeClient  # type: ignore[attr-defined]
    fake_exc = types.ModuleType("mlflow.exceptions")
    fake_exc.RestException = FakeRestException  # type: ignore[attr-defined]
    fake.exceptions = fake_exc  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.exceptions", fake_exc)
    return recorded


def _run(run_id: str = "abc123") -> types.SimpleNamespace:
    info = types.SimpleNamespace(run_id=run_id, artifact_uri=f"mlflow-artifacts:/1/{run_id}")
    return types.SimpleNamespace(info=info)


def test_registers_bare_file_source_and_returns_version(registry_calls: dict) -> None:
    version = _register_model(_run(), "aerial-object-detector")  # type: ignore[arg-type]
    assert version == "1"
    name, source, run_id = registry_calls["version_args"]
    assert name == "aerial-object-detector"
    assert source == "mlflow-artifacts:/1/abc123/weights/best.pt"  # the raw .pt file, not a dir
    assert run_id == "abc123"


def test_duplicate_registered_model_is_suppressed(registry_calls: dict) -> None:
    assert _register_model(_run(), "m") == "1"  # type: ignore[arg-type]
    assert _register_model(_run(), "m") == "2"  # type: ignore[arg-type]
    assert registry_calls["models_created"] == 2
    assert registry_calls["versions_created"] == 2
