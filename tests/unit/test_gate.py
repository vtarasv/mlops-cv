"""Unit tests for the promotion gate (pure; no mlflow): absolute floors + champion/challenger."""

from __future__ import annotations

import pytest

from mlops_cv.config import Settings
from mlops_cv.evaluation.gate import (
    GateThresholds,
    evaluate_gate,
    fetch_champion_metrics,
    promote,
)
from mlops_cv.startup import StartupError


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


# --- the champion's headline ---


def test_champion_metrics_are_read_off_its_training_run(registry) -> None:
    registry.promote(version="7", run_id="train-7")
    registry.runs["train-7"].data.metrics["test/mAP50-95"] = 0.41

    assert fetch_champion_metrics(Settings(), "server", registry=registry) == {
        "test/mAP50-95": 0.41
    }


def test_no_champion_is_the_bootstrap_answer(registry) -> None:
    """The one consumer for which an empty registry is an answer, not a refusal."""
    assert fetch_champion_metrics(Settings(), "server", registry=registry) is None


def test_a_champion_whose_training_run_is_gone_is_not_mistaken_for_no_champion(registry) -> None:
    """Reading a pruned run as "no champion" would hand the challenger a silent bootstrap win."""
    registry.promote(version="7", run_id="train-7")
    del registry.runs["train-7"]

    with pytest.raises(StartupError, match="train-7"):
        fetch_champion_metrics(Settings(), "server", registry=registry)


# --- promotion ---

# Each target needs only its own served tag, at the version's training imgsz (1280).
SERVED = {
    "server": {"optimize.onnx_1280": "runs:/record-run/onnx/best_1280.onnx"},
    "edge": {"optimize.ncnn_fp16_1280": "runs:/record-run/ncnn/ncnn-fp16-1280_ncnn_model"},
}


def _challenger(registry, tags=SERVED["server"], gate: str | None = "True") -> None:
    """v7 on its own training run, which evaluation tagged with ``gate``; v6 is the champion."""
    registry.promote(version="6")
    registry.add_run(
        "train-7", tags={} if gate is None else {"gate.passed": gate}, params={"imgsz": "1280"}
    )
    registry.add_version(version="7", run_id="train-7", tags=tags)


@pytest.mark.parametrize("target", sorted(SERVED))
def test_promotion_refuses_a_version_missing_its_targets_served_tag(registry, target) -> None:
    """A champion without its served artifact would stop its services starting."""
    other = "edge" if target == "server" else "server"
    _challenger(registry, tags=SERVED[other])

    with pytest.raises(StartupError, match=next(iter(SERVED[target]))):
        promote(Settings(), "7", target, registry=registry)
    assert registry.aliases["champion"] == "6"


@pytest.mark.parametrize("target", sorted(SERVED))
def test_promotion_needs_only_the_targets_own_served_tag(registry, target) -> None:
    """No other target's artifact and no device measurement: the Pi is never on this path."""
    _challenger(registry, tags=SERVED[target])

    promote(Settings(), "7", target, registry=registry)

    assert registry.aliases["champion"] == "7"


def test_promotion_reads_the_served_tag_at_the_training_imgsz(registry) -> None:
    """A version trained at 1280 is not servable through a 640 graph."""
    _challenger(registry, tags={"optimize.onnx_640": "runs:/record-run/onnx/best_640.onnx"})

    with pytest.raises(StartupError, match="optimize.onnx_1280"):
        promote(Settings(), "7", "server", registry=registry)


@pytest.mark.parametrize("gate", ["False", None])
def test_promotion_refuses_a_version_without_a_passing_gate(registry, gate: str | None) -> None:
    """The verdict evaluation left on the training run is what promotion acts on."""
    _challenger(registry, gate=gate)

    with pytest.raises(StartupError, match="gate"):
        promote(Settings(), "7", "server", registry=registry)
    assert registry.aliases["champion"] == "6"


def test_promotion_moves_the_alias_to_a_servable_gate_winner(registry) -> None:
    _challenger(registry)

    promote(Settings(), "7", "server", registry=registry)

    assert registry.aliases["champion"] == "7"
