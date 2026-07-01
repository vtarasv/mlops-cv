"""Unit tests for ``evaluate._promotion_version`` branch/parse logic, mlflow faked (CI-safe)."""

from __future__ import annotations

import sys
import types

import pytest

from mlops_cv.eval.evaluate import _promotion_version

NAME = "aerial-object-detector"


class _FakeVersion:
    def __init__(self, version: str) -> None:
        self.version = version


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Fake ``mlflow``; the ``MlflowClient`` records calls; ``search_result`` is tunable."""
    recorded: dict = {"search_result": [_FakeVersion("5")]}

    class FakeClient:
        def get_model_version_by_alias(self, name: str, alias: str) -> _FakeVersion:
            recorded["alias"] = (name, alias)
            return _FakeVersion("9")

        def search_model_versions(self, filter_string: str) -> list:
            recorded["search"] = filter_string
            return recorded["search_result"]

    fake = types.ModuleType("mlflow")
    fake.MlflowClient = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return recorded


def test_alias_uri_returns_version_from_alias_lookup(calls: dict) -> None:
    assert _promotion_version(f"models:/{NAME}@champion", NAME) == "9"
    assert calls["alias"] == (NAME, "champion")


def test_versioned_uri_is_parsed_without_registry_query(calls: dict) -> None:
    assert _promotion_version(f"models:/{NAME}/7", NAME) == "7"
    assert "alias" not in calls and "search" not in calls  # pure string parse


def test_run_uri_searches_by_run_id_and_name(calls: dict) -> None:
    assert _promotion_version("runs:/abc123/weights/best.pt", NAME) == "5"  # found[0].version
    assert calls["search"] == f"run_id='abc123' and name='{NAME}'"


def test_run_uri_returns_none_when_nothing_registered(calls: dict) -> None:
    calls["search_result"] = []
    assert _promotion_version("runs:/abc123/weights/best.pt", NAME) is None


def test_local_path_returns_none_without_registry_query(calls: dict) -> None:
    assert _promotion_version("weights/best.pt", NAME) is None
    assert "alias" not in calls and "search" not in calls


def test_bare_models_name_returns_none(calls: dict) -> None:
    # ``models:/name`` (no @, no /) matches neither sub-branch -> None, no registry query.
    assert _promotion_version(f"models:/{NAME}", NAME) is None
    assert "alias" not in calls and "search" not in calls
