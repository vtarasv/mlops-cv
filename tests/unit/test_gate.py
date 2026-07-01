"""Unit tests for the promotion gate (pure; no mlflow): absolute floors + champion/challenger."""

from __future__ import annotations

from mlops_cv.eval.gate import GateThresholds, evaluate_gate


def _metrics(
    primary: float = 0.5, m50: float = 0.6, p: float = 0.8, r: float = 0.7, prefix: str = "test"
) -> dict[str, float]:
    return {
        f"{prefix}/mAP50-95": primary,
        f"{prefix}/mAP50": m50,
        f"{prefix}/precision": p,
        f"{prefix}/recall": r,
    }


def test_bootstrap_passes_when_floors_met() -> None:
    g = evaluate_gate(_metrics(), GateThresholds(min_map50_95=0.3), None)
    assert g.passed
    assert g.is_challenger_win
    assert g.champion_primary is None


def test_bootstrap_fails_below_floor() -> None:
    g = evaluate_gate(_metrics(primary=0.2), GateThresholds(min_map50_95=0.3), None)
    assert not g.passed  # floor fails even though there's no champion to beat
    assert g.is_challenger_win


def test_challenger_beats_champion() -> None:
    g = evaluate_gate(_metrics(primary=0.5), GateThresholds(), _metrics(primary=0.4))
    assert g.passed
    assert g.is_challenger_win
    assert g.champion_primary == 0.4


def test_challenger_worse_than_champion_fails() -> None:
    g = evaluate_gate(_metrics(primary=0.3), GateThresholds(), _metrics(primary=0.4))
    assert not g.passed
    assert not g.is_challenger_win


def test_challenger_equal_with_zero_margin_passes() -> None:
    g = evaluate_gate(
        _metrics(primary=0.4), GateThresholds(min_improvement=0.0), _metrics(primary=0.4)
    )
    assert g.is_challenger_win and g.passed  # boundary is inclusive (>=)


def test_default_margin_requires_improvement() -> None:
    # GateThresholds() carries a 0.01 default margin: an equal-metric challenger fails.
    assert GateThresholds().min_improvement == 0.01
    g = evaluate_gate(_metrics(primary=0.4), GateThresholds(), _metrics(primary=0.4))
    assert not g.is_challenger_win  # 0.4 < 0.4 + 0.01 (default margin)
    assert not g.passed


def test_challenger_within_margin_fails() -> None:
    g = evaluate_gate(
        _metrics(primary=0.42), GateThresholds(min_improvement=0.05), _metrics(primary=0.4)
    )
    assert not g.is_challenger_win  # 0.42 < 0.4 + 0.05


def test_absolute_floor_boundary_inclusive() -> None:
    g = evaluate_gate(_metrics(primary=0.3), GateThresholds(min_map50_95=0.3), None)
    assert g.passed  # value == threshold passes


def test_champion_missing_primary_is_bootstrap() -> None:
    g = evaluate_gate(_metrics(), GateThresholds(), {"test/precision": 0.99})  # no mAP50-95 key
    assert g.is_challenger_win
    assert g.champion_primary is None


def test_summary_mentions_pass_and_fail() -> None:
    assert evaluate_gate(_metrics(), GateThresholds(), None).summary().startswith("GATE PASS")
    failing = evaluate_gate(_metrics(primary=0.0), GateThresholds(min_map50_95=0.9), None)
    assert failing.summary().startswith("GATE FAIL")
