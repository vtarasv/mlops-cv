"""Unit tests for the pure per-class metric mapping (no GPU, no mlflow, no numpy)."""

from __future__ import annotations

from mlops_cv.training.callbacks import per_class_metrics

NAMES = {0: "person", 1: "vehicle", 2: "two-three-wheeler"}


def test_only_present_classes_emitted() -> None:
    # maps is class-id-indexed; ultralytics back-fills absent classes with the overall mAP, so the
    # value at index 1 (vehicle, not in ap_class_index) must be ignored, not logged.
    maps = [0.5, 0.99, 0.3]  # index 1 == 0.99 is the back-filled overall mAP
    out = per_class_metrics(maps, [0, 2], NAMES)
    assert out == {"metrics/mAP50-95/person": 0.5, "metrics/mAP50-95/two-three-wheeler": 0.3}


def test_prefix_override() -> None:
    out = per_class_metrics([0.4, 0.0, 0.0], [0], NAMES, prefix="test/mAP50-95")
    assert out == {"test/mAP50-95/person": 0.4}


def test_values_coerced_to_float() -> None:
    out = per_class_metrics([1, 0, 0], [0], NAMES)  # int in -> float out
    assert out == {"metrics/mAP50-95/person": 1.0}
    assert isinstance(out["metrics/mAP50-95/person"], float)
