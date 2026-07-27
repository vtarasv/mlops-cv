"""Unit tests for ``tracking.resolve.resolve_model`` URI/branch/slug logic, mlflow faked."""

from __future__ import annotations

import sys
import types

import pytest

from mlops_cv.tracking.resolve import resolve_model


class _FakeVersion:
    source = "mlflow-artifacts:/1/run/artifacts/weights/best.pt"
    run_id = "run"


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Inject fake ``mlflow`` modules and record the calls the resolver makes."""
    recorded: dict = {}

    class FakeClient:
        def get_model_version_by_alias(self, name: str, alias: str) -> _FakeVersion:
            recorded["alias"] = (name, alias)
            return _FakeVersion()

        def get_model_version(self, name: str, version: str) -> _FakeVersion:
            recorded["version"] = (name, version)
            return _FakeVersion()

    def _download(artifact_uri: str) -> str:
        recorded["download_uri"] = artifact_uri
        return "/tmp/dl/best.pt"

    fake = types.ModuleType("mlflow")
    fake.MlflowClient = FakeClient  # type: ignore[attr-defined]
    artifacts = types.ModuleType("mlflow.artifacts")
    artifacts.download_artifacts = _download  # type: ignore[attr-defined]
    fake.artifacts = artifacts  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.artifacts", artifacts)
    return recorded


def test_alias_resolves_and_downloads_version_source(calls: dict) -> None:
    path, slug = resolve_model("models:/aerial-object-detector@champion")
    assert calls["alias"] == ("aerial-object-detector", "champion")
    assert calls["download_uri"] == _FakeVersion.source  # the version's own source, not a run path
    assert slug == "aerial-object-detector-champion"
    assert path.name == "best.pt"


def test_version_resolves_and_downloads_version_source(calls: dict) -> None:
    resolve_model("models:/aerial-object-detector/7")
    assert calls["version"] == ("aerial-object-detector", "7")
    assert calls["download_uri"] == _FakeVersion.source


def test_run_uri_passes_through_without_registry_lookup(calls: dict) -> None:
    _, slug = resolve_model("runs:/abc123/weights/best.pt")
    assert calls["download_uri"] == "runs:/abc123/weights/best.pt"
    assert "alias" not in calls and "version" not in calls
    assert slug == "abc123-weights-best.pt"
