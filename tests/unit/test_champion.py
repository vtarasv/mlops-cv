"""Champion resolution: alias -> model version -> training run, every dead end a named refusal."""

from __future__ import annotations

import os
import sys
import types

import pytest

from mlops_cv.config import Settings, load_settings
from mlops_cv.startup import StartupError
from mlops_cv.tracking import champion

NAME = "aerial-object-detector"


# --- the alias ---


def test_resolves_the_alias_to_the_promoted_version(registry) -> None:
    registry.promote(version="9", run_id="train-9")

    found = champion.resolve(Settings(), registry=registry)

    assert found.version == "9"
    assert registry.asked == [("alias", NAME, "champion")]


def test_resolves_a_pinned_version_number_without_touching_the_alias(registry) -> None:
    """The on-device harness measures a named version; the alias may point elsewhere by then."""
    registry.promote(version="9")
    registry.add_version(version="7")

    found = champion.resolve(Settings(), "7", registry=registry)

    assert found.version == "7"
    assert registry.asked == [("version", NAME, "7")]


def test_no_champion_is_a_named_refusal_not_a_traceback(registry) -> None:
    """The ordinary state of a fresh install."""
    with pytest.raises(StartupError) as exc:
        champion.resolve(Settings(), registry=registry)
    message = str(exc.value)
    assert "champion" in message  # which alias
    assert NAME in message  # on which model
    assert "make train" in message  # and how to get one


def test_a_pinned_version_that_does_not_exist_is_a_named_refusal(registry) -> None:
    registry.promote(version="9")
    with pytest.raises(StartupError) as exc:
        champion.resolve(Settings(), "7", registry=registry)
    assert "7" in str(exc.value)


# --- the training run ---


def test_training_run_is_the_run_that_produced_the_version(registry) -> None:
    version = registry.promote(version="9", run_id="train-9")
    registry.runs["train-9"].data.metrics["test/mAP50-95"] = 0.41

    run = champion.training_run(version, registry=registry)

    assert run.info.run_id == "train-9"
    assert run.data.metrics["test/mAP50-95"] == 0.41


def test_a_version_naming_no_training_run_is_a_named_refusal(registry) -> None:
    """A version registered outside a tracked run: the walk has no second step to take."""
    version = registry.promote(version="8", run_id=None)
    with pytest.raises(StartupError) as exc:
        champion.training_run(version, registry=registry)
    message = str(exc.value)
    assert "8" in message  # which model version
    assert "make train" in message


def test_a_training_run_that_is_gone_is_a_named_refusal(registry) -> None:
    """The price of a pointer: a pruned run must read as a refusal, not a RestException."""
    version = registry.promote(version="8", run_id="train-8")
    del registry.runs["train-8"]
    with pytest.raises(StartupError) as exc:
        champion.training_run(version, registry=registry)
    message = str(exc.value)
    assert "train-8" in message
    assert "make train" in message


# --- the session ---


def test_resolving_without_an_injected_registry_configures_the_session_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one ordering invariant of the walk, owned here: the tracking URI is exported before
    the client that reads it from the environment is built."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW__TRACKING_URI", "http://host:1234")
    seen: dict[str, str | None] = {}

    class FakeClient:
        def __init__(self) -> None:
            seen["env_at_construction"] = os.environ.get("MLFLOW_TRACKING_URI")

        def get_model_version_by_alias(self, name: str, alias: str) -> types.SimpleNamespace:
            return types.SimpleNamespace(version="1", run_id="r", tags={})

    fake = types.ModuleType("mlflow")
    fake.MlflowClient = FakeClient  # type: ignore[attr-defined]
    exceptions = types.ModuleType("mlflow.exceptions")
    exceptions.MlflowException = type("MlflowException", (Exception,), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.exceptions", exceptions)

    found = champion.resolve(load_settings(base_dir="/nonexistent"))

    assert found.version == "1"
    assert seen["env_at_construction"] == "http://host:1234"
