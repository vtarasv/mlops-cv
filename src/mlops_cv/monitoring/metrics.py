"""What the monitor exposes: the readings Prometheus scrapes."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter, Gauge

if TYPE_CHECKING:
    from mlops_cv.monitoring.drift import WindowVerdict
    from mlops_cv.monitoring.monitor import PredictionHealth


class MonitorMetrics:
    """The monitor's collectors, and the only place that writes to them."""

    def __init__(
        self, registry: CollectorRegistry, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.registry = registry
        self._clock = clock
        self._last_window = clock()
        self._last_prediction = clock()
        self._seen_classes: set[str] = set()

        self._score = Gauge(
            "drift_score",
            "Window distance from the training-scene cloud, in standard deviations of it",
            ["statistic"],
            registry=registry,
        )
        self._threshold = Gauge(
            "drift_threshold",
            "Derived bar: the widest distance any training scene itself reaches",
            ["statistic"],
            registry=registry,
        )
        self._windows = Counter(
            "drift_windows", "Completed drift windows, by verdict", ["verdict"], registry=registry
        )
        self._episodes = Counter(
            "drift_episodes", "Drift episodes opened by the episode rule", registry=registry
        )
        self._skipped = Counter(
            "drift_frames_skipped", "Frames that could not be profiled", registry=registry
        )
        self._unrecorded = Counter(
            "drift_episodes_unrecorded",
            "Episodes that fired but could not be recorded as evidence",
            registry=registry,
        )
        self._age = Gauge(
            "drift_window_age_seconds",
            "Seconds since the last completed drift window (since startup if there is none)",
            registry=registry,
        )
        self._age.set_function(lambda: self._clock() - self._last_window)

        self._detections = Gauge(
            "prediction_detections_per_frame",
            "Detections per frame over the last window of detection events (charted, never judged)",
            registry=registry,
        )
        self._confidence = Gauge(
            "prediction_mean_confidence",
            "Mean detection confidence over the last window of detection events",
            registry=registry,
        )
        self._class_share = Gauge(
            "prediction_class_share",
            "Share of detections per class over the last window of detection events",
            ["cls"],
            registry=registry,
        )
        self._prediction_age = Gauge(
            "prediction_window_age_seconds",
            "Seconds since the last window of detection events (since startup if there is none)",
            registry=registry,
        )
        self._prediction_age.set_function(lambda: self._clock() - self._last_prediction)

    def observe_window(self, verdict: WindowVerdict) -> None:
        """Publish one completed window: every score beside its own bar, and the verdict."""
        for statistic, score in verdict.scores.items():
            self._score.labels(statistic).set(score)
            self._threshold.labels(statistic).set(verdict.thresholds[statistic])
        self._windows.labels("drifted" if verdict.drifted else "ok").inc()
        self._last_window = self._clock()

    def observe_episode(self) -> None:
        self._episodes.inc()

    def observe_skipped_frame(self) -> None:
        self._skipped.inc()

    def observe_unrecorded_episode(self) -> None:
        self._unrecorded.inc()

    def observe_prediction(self, health: PredictionHealth) -> None:
        """Publish one window of prediction health; a class that stopped appearing reads zero."""
        self._detections.set(health.detections_per_frame)
        self._confidence.set(health.mean_confidence)
        for absent in self._seen_classes - set(health.class_shares):
            self._class_share.labels(absent).set(0.0)  # a stale share reads as a live one
        for cls, share in health.class_shares.items():
            self._class_share.labels(cls).set(share)
            self._seen_classes.add(cls)
        self._last_prediction = self._clock()
