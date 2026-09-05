"""Record runs: one child per producer under a training run, found or created, never held open."""

from __future__ import annotations

import pytest

from mlops_cv.tracking.records import PARENT_TAG, open_record, tag_filter

PARENT = "train-1"
OPTIMIZE = {"optimize.model": "models:/m@champion", "optimize.split": "test"}
MONITOR = {"monitor.model": "m", "monitor.version": "8"}


def test_a_record_is_created_as_a_terminated_child_of_its_parent(registry) -> None:
    """SIGTERM runs no atexit handler, so a held-open run would strand RUNNING on the first stop."""
    registry.add_run(PARENT)

    run_id = open_record(registry, PARENT, "optimize.model", run_name="optimize", tags=OPTIMIZE)

    run = registry.runs[run_id]
    assert run.data.tags[PARENT_TAG] == PARENT
    assert run.data.tags["mlflow.runName"] == "optimize"
    assert run.data.tags["optimize.split"] == "test"
    assert run.info.experiment_id == registry.runs[PARENT].info.experiment_id  # nested = same
    assert run.info.status == "FINISHED"


def test_a_second_opening_finds_the_record_instead_of_adding_a_sibling(registry) -> None:
    """Two children holding two half-comparable ladders is worse than one appended in place."""
    registry.add_run(PARENT)

    first = open_record(registry, PARENT, "optimize.model", run_name="optimize", tags=OPTIMIZE)
    second = open_record(registry, PARENT, "optimize.model", run_name="optimize", tags=OPTIMIZE)

    assert second == first
    assert len(registry.runs) == 2  # the parent and one record


def test_records_of_another_kind_under_the_same_parent_are_not_confused(registry) -> None:
    """Both are children of the same parent; only the marker tag tells them apart."""
    registry.add_run(PARENT)

    optimize = open_record(registry, PARENT, "optimize.model", run_name="optimize", tags=OPTIMIZE)
    monitor = open_record(registry, PARENT, "monitor.model", run_name="monitor", tags=MONITOR)

    assert optimize != monitor
    again = open_record(registry, PARENT, "optimize.model", run_name="optimize", tags=OPTIMIZE)
    assert again == optimize
    again = open_record(registry, PARENT, "monitor.model", run_name="monitor", tags=MONITOR)
    assert again == monitor


def test_children_that_are_no_record_are_ignored(registry) -> None:
    """A training run may carry other children (an HPO trial); only a record may be written to."""
    registry.add_run(PARENT)
    registry.add_run("trial-3", tags={PARENT_TAG: PARENT, "mlflow.runName": "hpo-trial-3"})

    record = open_record(registry, PARENT, "optimize.model", run_name="optimize", tags=OPTIMIZE)

    assert record != "trial-3"
    assert len(registry.runs) == 3  # parent, trial, and a fresh record


def test_the_marker_must_be_among_the_tags(registry) -> None:
    """The marker is the record's identity; a record created without it could never be found."""
    registry.add_run(PARENT)
    with pytest.raises(ValueError, match="optimize.model"):
        open_record(registry, PARENT, "optimize.model", run_name="optimize", tags={"x": "y"})


def test_tag_filter_spells_the_grammar_the_server_parses() -> None:
    """One spelling for every tag search — the fakes above parse exactly this shape."""
    assert tag_filter("data.stamp", "abc") == "tags.data.stamp = 'abc'"
    assert tag_filter(PARENT_TAG, PARENT) == f"tags.mlflow.parentRunId = '{PARENT}'"
