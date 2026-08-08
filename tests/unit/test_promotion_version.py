"""Unit tests for ``tracking.resolve``'s registered-version seam — ``model_version`` (the full
entity), ``registered_version`` (its number), ``run_id_from_uri`` — mlflow faked."""

from __future__ import annotations

import sys
import types

import pytest

from mlops_cv.tracking.resolve import model_version, registered_version, run_id_from_uri

NAME = "aerial-object-detector"


class _FakeVersion:
    def __init__(self, version: str, run_id: str = "run") -> None:
        self.version = version
        self.run_id = run_id


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Fake ``mlflow``; the ``MlflowClient`` records calls; ``search_result`` is tunable."""
    recorded: dict = {"search_result": [_FakeVersion("5")]}

    class FakeClient:
        def get_model_version_by_alias(self, name: str, alias: str) -> _FakeVersion:
            recorded["alias"] = (name, alias)
            return _FakeVersion("9")

        def get_model_version(self, name: str, version: str) -> _FakeVersion:
            recorded["version"] = (name, version)
            return _FakeVersion(version)

        def search_model_versions(self, filter_string: str) -> list:
            recorded["search"] = filter_string
            return recorded["search_result"]

    fake = types.ModuleType("mlflow")
    fake.MlflowClient = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return recorded


def test_alias_uri_returns_version_from_alias_lookup(calls: dict) -> None:
    assert registered_version(f"models:/{NAME}@champion", NAME) == "9"
    assert calls["alias"] == (NAME, "champion")


def test_versioned_uri_is_parsed_without_registry_query(calls: dict) -> None:
    assert registered_version(f"models:/{NAME}/7", NAME) == "7"
    assert "alias" not in calls and "search" not in calls  # pure string parse


def test_run_uri_searches_by_run_id_and_name(calls: dict) -> None:
    assert registered_version("runs:/abc123/weights/best.pt", NAME) == "5"  # found[0].version
    assert calls["search"] == f"run_id='abc123' and name='{NAME}'"


def test_run_uri_returns_none_when_nothing_registered(calls: dict) -> None:
    calls["search_result"] = []
    assert registered_version("runs:/abc123/weights/best.pt", NAME) is None


def test_local_path_returns_none_without_registry_query(calls: dict) -> None:
    assert registered_version("weights/best.pt", NAME) is None
    assert "alias" not in calls and "search" not in calls


def test_bare_models_name_returns_none(calls: dict) -> None:
    # ``models:/name`` (no @, no /) matches neither sub-branch -> None, no registry query.
    assert registered_version(f"models:/{NAME}", NAME) is None
    assert "alias" not in calls and "search" not in calls


def test_model_version_returns_the_full_entity_for_an_alias(calls: dict) -> None:
    """Not just the number: callers read tags (artifact addresses) and run_id off the entity."""
    version = model_version(f"models:/{NAME}@champion", NAME)
    assert calls["alias"] == (NAME, "champion")
    assert (version.version, version.run_id) == ("9", "run")  # type: ignore[attr-defined]


def test_model_version_fetches_a_versioned_ref_from_the_registry(calls: dict) -> None:
    assert model_version(f"models:/{NAME}/7", NAME).version == "7"  # type: ignore[attr-defined]
    assert calls["version"] == (NAME, "7")


def test_model_version_searches_run_uris_by_run_id_and_name(calls: dict) -> None:
    assert model_version("runs:/abc123/weights/best.pt", NAME).version == "5"  # type: ignore[attr-defined]
    assert calls["search"] == f"run_id='abc123' and name='{NAME}'"


def test_model_version_local_path_maps_to_no_version(calls: dict) -> None:
    assert model_version("weights/best.pt", NAME) is None
    assert "alias" not in calls and "search" not in calls and "version" not in calls


def test_run_id_from_uri_reads_runs_uris_and_nothing_else() -> None:
    assert run_id_from_uri("runs:/abc123/weights/best.pt") == "abc123"
    assert run_id_from_uri("runs:/abc123") == "abc123"
    assert run_id_from_uri(f"models:/{NAME}@champion") is None
    assert run_id_from_uri("s3://bucket/some/path") is None
