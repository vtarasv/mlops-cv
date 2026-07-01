"""Unit tests for eval report formatting."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlops_cv.eval.report import (
    comparison_table_md,
    headline_metrics,
    metrics_csv_rows,
    percentiles,
    write_report,
)


def _box(mp: float, mr: float, map50: float, map_: float) -> SimpleNamespace:
    """An ultralytics-``results.box`` stand-in exposing the four headline attributes."""
    return SimpleNamespace(mp=mp, mr=mr, map50=map50, map=map_)


def _metrics(p: float, r: float, m50: float, m: float, prefix: str = "test") -> dict[str, float]:
    return {
        f"{prefix}/precision": p,
        f"{prefix}/recall": r,
        f"{prefix}/mAP50": m50,
        f"{prefix}/mAP50-95": m,
    }


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


def test_percentiles_median() -> None:
    assert percentiles([10, 20, 30, 40, 50], (50,))[50.0] == pytest.approx(30.0)


def test_percentiles_linear_interpolation() -> None:
    out = percentiles(list(range(1, 101)), (50, 95))
    assert out[50.0] == pytest.approx(50.5)
    assert out[95.0] == pytest.approx(95.05)


def test_percentiles_empty_returns_zeros() -> None:
    assert percentiles([], (50, 95)) == {50.0: 0.0, 95.0: 0.0}


def test_comparison_table_no_champion_is_two_columns() -> None:
    md = comparison_table_md(_metrics(0.8, 0.7, 0.6, 0.5), None)
    assert "Candidate" in md
    assert "Champion" not in md
    assert "precision" in md and "mAP50-95" in md


def test_comparison_table_with_champion_has_delta() -> None:
    md = comparison_table_md(_metrics(0.8, 0.7, 0.6, 0.5), _metrics(0.6, 0.6, 0.5, 0.4))
    assert "Champion" in md and "Δ" in md
    assert "+0.1000" in md  # mAP50-95 delta = 0.5 - 0.4


def test_metrics_csv_rows_candidate_only() -> None:
    rows = metrics_csv_rows(_metrics(0.8, 0.7, 0.6, 0.5), None)
    assert len(rows) == 1 and rows[0]["role"] == "candidate"


def test_metrics_csv_rows_with_champion() -> None:
    rows = metrics_csv_rows(_metrics(0.8, 0.7, 0.6, 0.5), _metrics(0.6, 0.6, 0.5, 0.4))
    assert [r["role"] for r in rows] == ["candidate", "champion"]


def test_write_report_creates_md_and_csv(tmp_path: Path) -> None:
    candidate = _metrics(0.8, 0.7, 0.6, 0.5) | {"test/mAP50-95/person": 0.55}
    gate = SimpleNamespace(summary=lambda: "GATE PASS: test/mAP50-95=0.5000", checks=[])
    md, csv_path = write_report(
        tmp_path,
        candidate=candidate,
        champion=None,
        gate=gate,  # type: ignore
        latency={"latency/p50_ms": 12.3},
    )
    assert md.exists() and csv_path.exists()
    text = md.read_text(encoding="utf-8")
    assert "GATE" in text
    assert "person" in text  # the per-class mAP50-95 row
    assert "latency/p50_ms" in text
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["role"] == "candidate"
    assert rows[0]["test/mAP50-95"] == "0.5"
