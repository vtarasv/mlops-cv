"""Optimize driver's pure helpers: artifact addressing and record-run reuse."""

from __future__ import annotations

from types import SimpleNamespace

from mlops_cv.optimize.optimize import (
    RECORD_TAG,
    existing_record_run,
    record_tags,
)


def _run(run_id: str, **tags: str) -> SimpleNamespace:
    """An mlflow ``Run`` stand-in exposing the two fields the lookup reads."""
    return SimpleNamespace(info=SimpleNamespace(run_id=run_id), data=SimpleNamespace(tags=tags))


class _Client:
    """Records the search it was asked for and replays a fixed list of children."""

    def __init__(self, children: list[SimpleNamespace]) -> None:
        self.children = children
        self.filter_string = ""

    def search_runs(self, experiment_ids, filter_string):
        self.experiment_ids = experiment_ids
        self.filter_string = filter_string
        return self.children


def test_reuses_an_earlier_record_instead_of_adding_a_sibling() -> None:
    """Two children holding two half-comparable ladders is worse than one overwritten in place."""
    client = _Client([_run("child-1", **{RECORD_TAG: "models:-m-1"})])
    assert existing_record_run(client, "parent-1", "0") == "child-1"
    assert "parent-1" in client.filter_string
    assert client.experiment_ids == ["0"]


def test_children_that_are_not_optimization_records_are_ignored() -> None:
    """A training run may carry other children; only an optimization record may be written to."""
    client = _Client([_run("some-other-child", **{"mlflow.runName": "hpo-trial-3"})])
    assert existing_record_run(client, "parent-1", "0") is None


def test_a_run_with_no_children_starts_a_fresh_record() -> None:
    assert existing_record_run(_Client([]), "parent-1", "0") is None


def test_record_tags_pin_the_resolved_model_not_just_the_reference() -> None:
    """An alias names whatever is champion *now*; the record must survive the alias moving."""
    assert record_tags("aerial-object-detector-champion", "test", "4", "train-run-1") == {
        RECORD_TAG: "aerial-object-detector-champion",
        "optimize.split": "test",
        "optimize.version": "4",
        "optimize.source_run": "train-run-1",
    }


def test_record_tags_omit_provenance_that_does_not_exist() -> None:
    """An ad-hoc model has no registered version — an empty tag would read as a real answer."""
    assert record_tags("some-local-weights.pt", "test", None, None) == {
        RECORD_TAG: "some-local-weights.pt",
        "optimize.split": "test",
    }
