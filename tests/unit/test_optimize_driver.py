"""Optimize driver's pure helpers: the record's provenance tags (the record itself is
``tracking.records``' business, tested there)."""

from __future__ import annotations

from mlops_cv.optimize.optimize import RECORD_TAG, record_tags
from mlops_cv.tracking.records import open_record


def test_the_marker_is_among_the_record_tags_so_the_record_can_be_reopened(registry) -> None:
    """The driver's tags must satisfy the ledger's identity rule, or every run adds a sibling."""
    registry.add_run("train-run-1")
    tags = record_tags("models:/m@champion", "test", "4", "train-run-1")
    first = open_record(registry, "train-run-1", RECORD_TAG, run_name="optimize", tags=tags)
    second = open_record(registry, "train-run-1", RECORD_TAG, run_name="optimize", tags=tags)
    assert second == first


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
