"""The promotion gate: absolute metric floors **plus** champion/challenger vs the registry
champion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mlops_cv.tracking.metric_keys import PRIMARY, metric_key

if TYPE_CHECKING:
    from mlflow.entities.model_registry import ModelVersion


@dataclass(frozen=True)
class GateThresholds:
    """Absolute metric floors (default ``0.0``) + the challenger margin (default ``0.01``)."""

    min_map50_95: float = 0.0
    min_map50: float = 0.0
    min_precision: float = 0.0
    min_recall: float = 0.0
    min_improvement: float = 0.01  # challenger must beat the champion's primary by >= this


@dataclass(frozen=True)
class GateCheck:
    """One named pass/fail check, retained for the report table."""

    name: str
    value: float
    threshold: float
    passed: bool


@dataclass(frozen=True)
class GateResult:
    """The gate decision: ``passed`` is all absolute floors **and** the challenger comparison."""

    passed: bool
    checks: list[GateCheck]
    is_challenger_win: bool
    primary: str
    candidate_primary: float
    champion_primary: float | None

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        if self.champion_primary is None:
            comp = f"{self.primary}={self.candidate_primary:.4f} (no champion — bootstrap win)"
        else:
            comp = (
                f"{self.primary}={self.candidate_primary:.4f} "
                f"vs champion {self.champion_primary:.4f}"
            )
        failed = [c.name for c in self.checks if not c.passed]
        tail = "" if self.passed else f"; failed: {', '.join(failed)}"
        return f"GATE {verdict}: {comp}{tail}"


def evaluate_gate(
    metrics: Mapping[str, float],
    thresholds: GateThresholds,
    champion_metrics: Mapping[str, float] | None,
    *,
    prefix: str = "test",
) -> GateResult:
    """Decide pass/fail from already-computed metric dicts.

    Two parts:
    (1) absolute floors — each ``min_*`` against ``metrics[f"{prefix}/<name>"]`` (``>=``,
    boundary inclusive);
    (2) champion/challenger — on the primary key (``prefix`` + :data:`PRIMARY`) — if
    ``champion_metrics`` is ``None`` or lacks it this is a bootstrap win, else the candidate wins
    when its primary ``>=`` the champion's plus ``min_improvement``. ``passed`` is all floors
    **and** the win.
    """
    primary = metric_key(prefix, PRIMARY)
    floor_specs: Sequence[tuple[str, str, float]] = (
        (PRIMARY, primary, thresholds.min_map50_95),
        ("mAP50", metric_key(prefix, "mAP50"), thresholds.min_map50),
        ("precision", metric_key(prefix, "precision"), thresholds.min_precision),
        ("recall", metric_key(prefix, "recall"), thresholds.min_recall),
    )
    checks = [
        GateCheck(name, float(metrics.get(key, 0.0)), thr, float(metrics.get(key, 0.0)) >= thr)
        for name, key, thr in floor_specs
    ]
    floors_pass = all(c.passed for c in checks)

    candidate_primary = float(metrics.get(primary, 0.0))
    champion_primary = (
        float(champion_metrics[primary])
        if champion_metrics is not None and primary in champion_metrics
        else None
    )
    if champion_primary is None:
        is_challenger_win = True
        challenger_threshold = 0.0
    else:
        challenger_threshold = champion_primary + thresholds.min_improvement
        is_challenger_win = candidate_primary >= challenger_threshold
    checks.append(
        GateCheck("challenger", candidate_primary, challenger_threshold, is_challenger_win)
    )

    return GateResult(
        passed=floors_pass and is_challenger_win,
        checks=checks,
        is_challenger_win=is_challenger_win,
        primary=primary,
        candidate_primary=candidate_primary,
        champion_primary=champion_primary,
    )


def fetch_champion_metrics(model_name: str, alias: str = "champion") -> dict[str, float] | None:
    """The champion model version's logged run metrics, or ``None`` if no such alias/model exists.

    ``None`` means "no champion yet" → :func:`evaluate_gate` treats the candidate as a bootstrap.
    """
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = MlflowClient()
    try:
        version: ModelVersion = client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None
    if version is None or version.run_id is None:
        return None
    return dict(client.get_run(version.run_id).data.metrics)


def promote(model_name: str, version: str | int, alias: str = "champion") -> None:
    """Point the champion ``alias`` at ``version`` (MLflow 3 aliases replace deprecated stages)."""
    from mlflow import MlflowClient

    MlflowClient().set_registered_model_alias(model_name, alias, str(version))
