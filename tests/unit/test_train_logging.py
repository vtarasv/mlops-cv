"""Unit test for ``train._log_test_metrics`` wiring (a fake log sink)."""

from __future__ import annotations

from types import SimpleNamespace

from mlops_cv.training.train import _log_test_metrics


def test_logs_headline_and_present_class_metrics() -> None:
    logged: list[dict] = []
    fake_mlflow = SimpleNamespace(log_metrics=logged.append)
    results = SimpleNamespace(
        box=SimpleNamespace(mp=0.8, mr=0.7, map50=0.6, map=0.5),
        maps=[0.55, 0.99, 0.30],  # index 1 is the back-filled overall mAP -> must be ignored
        ap_class_index=[0, 2],
        names={0: "person", 1: "vehicle", 2: "two-three-wheeler"},
    )

    _log_test_metrics(fake_mlflow, results)

    assert logged[0] == {
        "test/precision": 0.8,
        "test/recall": 0.7,
        "test/mAP50": 0.6,
        "test/mAP50-95": 0.5,
    }
    assert logged[1] == {"test/mAP50-95/person": 0.55, "test/mAP50-95/two-three-wheeler": 0.30}
