"""Unit tests for the run-metric key vocabulary (pure; no ultralytics, no mlflow, no numpy).
"""

from __future__ import annotations

from types import SimpleNamespace

from mlops_cv.tracking.metric_keys import (
    HEADLINE_ATTRS,
    PRIMARY,
    TRAIN_PREFIX,
    headline_metrics,
    metric_key,
    per_class_metrics,
)

NAMES = {0: "person", 1: "vehicle", 2: "two-three-wheeler"}


def _box(mp: float, mr: float, map50: float, map_: float) -> SimpleNamespace:
    """An ultralytics-``results.box`` stand-in exposing the four headline attributes."""
    return SimpleNamespace(mp=mp, mr=mr, map50=map50, map=map_)


def test_golden_spellings() -> None:
    # The persisted contract: change any of these and previously recorded runs stop matching.
    assert PRIMARY == "mAP50-95"
    assert TRAIN_PREFIX == "metrics"
    assert metric_key("test", PRIMARY) == "test/mAP50-95"
    assert list(HEADLINE_ATTRS) == ["precision", "recall", "mAP50", "mAP50-95"]


def test_headline_metrics_keys_and_values() -> None:
    out = headline_metrics(_box(0.8, 0.7, 0.6, 0.5))
    assert out == {
        "test/precision": 0.8,
        "test/recall": 0.7,
        "test/mAP50": 0.6,
        "test/mAP50-95": 0.5,
    }


def test_headline_metrics_prefix_override() -> None:
    out = headline_metrics(_box(1, 1, 1, 1), prefix="val")
    assert set(out) == {"val/precision", "val/recall", "val/mAP50", "val/mAP50-95"}


def test_headline_metrics_coerces_float() -> None:
    out = headline_metrics(_box(1, 0, 0, 0))  # ints in
    assert out["test/precision"] == 1.0
    assert isinstance(out["test/precision"], float)


def test_per_class_only_present_classes_emitted() -> None:
    # maps is class-id-indexed; ultralytics back-fills absent classes with the overall mAP, so the
    # value at index 1 (vehicle, not in ap_class_index) must be ignored, not logged.
    maps = [0.5, 0.99, 0.3]  # index 1 == 0.99 is the back-filled overall mAP
    out = per_class_metrics(maps, [0, 2], NAMES)
    assert out == {"metrics/mAP50-95/person": 0.5, "metrics/mAP50-95/two-three-wheeler": 0.3}


def test_per_class_prefix_override() -> None:
    out = per_class_metrics([0.4, 0.0, 0.0], [0], NAMES, prefix="test/mAP50-95")
    assert out == {"test/mAP50-95/person": 0.4}


def test_per_class_values_coerced_to_float() -> None:
    out = per_class_metrics([1, 0, 0], [0], NAMES)  # int in -> float out
    assert out == {"metrics/mAP50-95/person": 1.0}
    assert isinstance(out["metrics/mAP50-95/person"], float)
