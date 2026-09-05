from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mlops_cv.pipelines.profiling import DRIFT_METRICS

if TYPE_CHECKING:
    from mlops_cv.pipelines.profiling import BaselineScene

# Ratio-scale statistics: distance from normal is multiplicative, so they are scored in log space.
LOG_SCALE_METRICS: tuple[str, ...] = ("blur",)

LOG_FLOOR = 1e-6


@dataclass(frozen=True)
class WindowVerdict:
    """One window's distance from normal: every score beside its own bar, and what crossed."""

    scores: dict[str, float]
    thresholds: dict[str, float]
    crossed: tuple[str, ...]

    @property
    def drifted(self) -> bool:
        """A window is drifted when **any single** statistic exceeds its threshold."""
        return bool(self.crossed)


@dataclass(frozen=True)
class DriftReference:
    """What normal looks like: the training-scene cloud, on the scale each statistic is scored."""

    center: dict[str, float]
    spread: dict[str, float]
    thresholds: dict[str, float]
    log_scale: tuple[str, ...]
    margin: float
    n_scenes: int

    @classmethod
    def from_scenes(
        cls,
        scenes: Sequence[BaselineScene],
        *,
        margin: float = 1.0,
        log_scale: tuple[str, ...] = LOG_SCALE_METRICS,
    ) -> DriftReference:
        """Derive the cloud and its thresholds from a published drift baseline."""
        if not scenes:
            raise ValueError("the drift baseline is empty — there are no training scenes to score")
        if margin <= 0:
            raise ValueError(f"the threshold margin must be positive, not {margin}")
        unknown = sorted(set(log_scale) - set(DRIFT_METRICS))
        if unknown:
            raise ValueError(
                f"{unknown} are not drift statistics, so naming them ratio-scale changes nothing; "
                f"the statistics scored are {list(DRIFT_METRICS)}"
            )
        center, spread, thresholds = {}, {}, {}
        for metric in DRIFT_METRICS:
            values = _scaled(
                np.array([scene.means[metric] for scene in scenes], dtype=float),
                metric in log_scale,
            )
            middle, sd = float(values.mean()), float(values.std())
            if sd == 0.0:
                raise ValueError(
                    f"the drift baseline's {metric!r} scene means have no spread — every distance "
                    "from this cloud is undefined, so no threshold can be derived from it"
                )
            center[metric], spread[metric] = middle, sd
            # The bar is the widest distance the training data itself reaches.
            thresholds[metric] = float(np.abs(values - middle).max() / sd) * margin
        return cls(
            center=center,
            spread=spread,
            thresholds=thresholds,
            log_scale=tuple(log_scale),
            margin=margin,
            n_scenes=len(scenes),
        )

    def judge(self, means: Mapping[str, float]) -> WindowVerdict:
        """Score one window's statistic means against the cloud and its thresholds."""
        missing = [metric for metric in DRIFT_METRICS if metric not in means]
        if missing:
            raise ValueError(
                f"the window carries no {missing} — a drift window is scored on "
                f"{list(DRIFT_METRICS)}"
            )
        scores = {metric: self._distance(metric, float(means[metric])) for metric in DRIFT_METRICS}
        crossed = tuple(
            metric for metric in DRIFT_METRICS if scores[metric] > self.thresholds[metric]
        )
        return WindowVerdict(scores=scores, thresholds=dict(self.thresholds), crossed=crossed)

    def _distance(self, metric: str, value: float) -> float:
        scaled = float(_scaled(np.array([value], dtype=float), metric in self.log_scale)[0])
        return abs(scaled - self.center[metric]) / self.spread[metric]


@dataclass
class EpisodeRule:
    """Fire-once episode semantics over consecutive drifted windows.

    Two consecutive drifted windows open an episode; it is reported once, stays silent while the
    condition holds, and re-arms only after a clean window.
    """

    consecutive: int = 2
    _run: int = 0  # consecutive drifted windows seen
    _active: bool = False  # an episode is open

    def observe(self, verdict: WindowVerdict) -> WindowVerdict | None:
        """Fold one window's verdict in; return the verdict that opens an episode, else ``None``."""
        if not verdict.drifted:
            self._run, self._active = 0, False
            return None
        self._run += 1
        if self._active or self._run < self.consecutive:
            return None
        self._active = True
        return verdict


def window_means(readings: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Mean of each drift statistic over one window of per-frame readings."""
    rows = list(readings)
    if not rows:
        raise ValueError("cannot take the mean of an empty window")
    return {
        metric: float(np.mean([float(row[metric]) for row in rows])) for metric in DRIFT_METRICS
    }


def _scaled(values: np.ndarray, log_scale: bool) -> np.ndarray:
    """Values on the scale their statistic is scored on."""
    return np.log(np.maximum(values, LOG_FLOOR)) if log_scale else values
