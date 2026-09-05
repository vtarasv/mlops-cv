"""The drift monitor: live frames in, drift readings out.

The reference is resolved **once at startup** (restart to pick up a promotion, like the
service and the consumers) and its thresholds are derived from the baseline there.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mlops_cv.config import Settings, get_settings
from mlops_cv.monitoring.drift import DriftReference, EpisodeRule, window_means
from mlops_cv.monitoring.metrics import MonitorMetrics
from mlops_cv.monitoring.resolve import REPROFILE_HINT, ResolvedBaseline, resolve_baseline
from mlops_cv.pipelines.profiling import profile_drift_bytes
from mlops_cv.startup import StartupError, exits_on_startup_error
from mlops_cv.streaming import consumer_parser
from mlops_cv.streaming.messages import (
    FRAME_MAX_MESSAGE_BYTES,
    DetectionEvent,
    FrameRef,
    InboundHeaders,
    parse_frame,
)

if TYPE_CHECKING:
    from mlops_cv.monitoring.drift import WindowVerdict
    from mlops_cv.pipelines.profiling import BaselineScene

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceMessage:
    """One message off the bus, reduced to what the monitor reads."""

    topic: str
    key: bytes | None = None
    value: bytes | None = None
    headers: InboundHeaders | None = None


# A poll: the next message, or ``None`` when the bus is quiet right now.
MessageSource = Callable[[], SourceMessage | None]


@dataclass(frozen=True)
class DriftWindow:
    """One completed window: its verdict, and the frames the verdict was computed from."""

    verdict: WindowVerdict
    frames: tuple[FrameRef, ...]

    def spans(self) -> list[dict[str, Any]]:
        """The window's footage as one index span per clip, in order of first appearance."""
        spans: dict[str, dict[str, Any]] = {}
        for frame in self.frames:
            span = spans.get(frame.sequence)
            if span is None:
                spans[frame.sequence] = {
                    "sequence": frame.sequence,
                    "first": frame.frame_index,
                    "last": frame.frame_index,
                    "n_frames": 1,
                }
                continue
            span["first"] = min(span["first"], frame.frame_index)
            span["last"] = max(span["last"], frame.frame_index)
            span["n_frames"] += 1
        return list(spans.values())

    def where(self) -> str:
        """The window's address, for a log line a reader can go and look at."""
        return ", ".join(f"{s['sequence']}/{s['first']}-{s['last']}" for s in self.spans())


# Where an opened episode goes; the default sink only logs.
EpisodeSink = Callable[[DriftWindow], None]


@dataclass(frozen=True)
class PredictionHealth:
    """One window of prediction-side readings — charted, never judged."""

    n_events: int
    detections_per_frame: float
    mean_confidence: float
    class_shares: dict[str, float]


def prediction_health(events: Sequence[DetectionEvent]) -> PredictionHealth:
    """Aggregate a window of Detection events into the prediction-health gauges."""
    if not events:
        raise ValueError("cannot summarize an empty window of detection events")
    boxes = [box for event in events for box in event.boxes]
    counts = Counter(box.cls for box in boxes)
    return PredictionHealth(
        n_events=len(events),
        detections_per_frame=len(boxes) / len(events),
        mean_confidence=sum(box.conf for box in boxes) / len(boxes) if boxes else 0.0,
        class_shares={cls: n / len(boxes) for cls, n in counts.items()},
    )


@dataclass
class _Window:
    """The tumbling window under construction: what has been sampled since the last verdict."""

    readings: list[dict[str, float]] = field(default_factory=list)
    frames: list[FrameRef] = field(default_factory=list)

    def add(self, reading: dict[str, float], ref: FrameRef) -> None:
        self.readings.append(reading)
        self.frames.append(ref)

    def close(self, reference: DriftReference) -> DriftWindow:
        """Judge the window and start the next one — tumbling, so no frame is judged twice."""
        window = DriftWindow(
            verdict=reference.judge(window_means(self.readings)), frames=tuple(self.frames)
        )
        self.readings.clear()
        self.frames.clear()
        return window


def drift_reference(scenes: Sequence[BaselineScene], settings: Settings) -> DriftReference:
    """Derive the thresholds the monitor judges against from the champion's own baseline."""
    try:
        return DriftReference.from_scenes(scenes, margin=settings.monitoring.threshold_margin)
    except ValueError as exc:
        raise StartupError(
            f"the champion's drift baseline yields no thresholds ({exc}) — there is nothing to "
            f"call normal. Fix: {REPROFILE_HINT}."
        ) from exc


def open_recorder(resolved: ResolvedBaseline) -> EpisodeSink:
    """The champion's episode ledger, or a sink that refuses with the reason it has none."""
    from mlops_cv.monitoring.evidence import open_record

    try:
        return open_record(resolved)
    except Exception as exc:
        # Bound to a name of its own: ``except ... as exc`` unbinds ``exc`` at the end of the
        # block, so the closure below would raise NameError instead of the reason.
        reason = exc
        logger.warning(
            f"drift episodes will not be recorded: the monitoring record could not be opened "
            f"({reason}). Scoring and metrics are unaffected; restart to retry."
        )

        def unavailable(window: DriftWindow) -> None:
            raise RuntimeError(f"the monitoring record was never opened: {reason}")

        return unavailable


def kafka_source(consumer, poll_timeout_s: float = 0.2) -> MessageSource:
    """The production seam: a subscribed consumer behind the loop's poll."""

    def poll() -> SourceMessage | None:
        msg = consumer.poll(poll_timeout_s)
        if msg is None:
            return None
        if msg.error():
            logger.warning(f"consumer error: {msg.error()}")
            return None
        return SourceMessage(
            topic=msg.topic(), key=msg.key(), value=msg.value(), headers=msg.headers()
        )

    return poll


def run_loop(
    settings: Settings,
    reference: DriftReference,
    source: MessageSource,
    metrics: MonitorMetrics,
    *,
    on_episode: EpisodeSink | None = None,
    max_messages: int | None = None,
    idle_timeout_s: float | None = None,
) -> dict[str, int]:
    """Score live windows against ``reference``; return counters when bounded.

    Runs forever unless ``max_messages`` / ``idle_timeout_s`` bounds it (tests, smokes).
    """
    monitoring, streaming = settings.monitoring, settings.streaming
    window = _Window()
    rule = EpisodeRule(consecutive=monitoring.consecutive_windows)
    health: list[DetectionEvent] = []
    # Prediction health windows over the same stretch of stream a drift window covers, so the
    # two panels beside each other read the same interval whatever the stride is.
    health_window = monitoring.window_frames * monitoring.sample_every
    counts = {
        "consumed": 0,
        "frames": 0,
        "sampled": 0,
        "events": 0,
        "windows": 0,
        "drifted": 0,
        "episodes": 0,
        "unrecorded": 0,  # episodes the sink could not record
        "skipped": 0,  # frames that could not be profiled
        "skipped_events": 0,  # detection events that could not be parsed
    }

    def handle_frame(msg: SourceMessage) -> None:
        counts["frames"] += 1
        if counts["frames"] % monitoring.sample_every:
            return  # sampling is not optional: profiling every frame might not keep up
        try:
            ref = parse_frame(msg.key, msg.headers)
            if msg.value is None:  # a tombstone carries no image
                raise ValueError("frame message has no value")
            reading = profile_drift_bytes(msg.value)
        except Exception as exc:
            counts["skipped"] += 1
            metrics.observe_skipped_frame()
            logger.warning(f"skipping unprofilable frame {msg.key!r}: {exc}")
            return
        counts["sampled"] += 1
        window.add(reading, ref)
        if len(window.readings) < monitoring.window_frames:
            return

        completed = window.close(reference)
        counts["windows"] += 1
        counts["drifted"] += int(completed.verdict.drifted)
        metrics.observe_window(completed.verdict)
        logger.info(f"window {completed.where()}: {_scoreline(completed.verdict)}")

        if rule.observe(completed.verdict) is None:
            return
        counts["episodes"] += 1
        metrics.observe_episode()
        logger.warning(
            f"DRIFT episode at {completed.where()}: "
            f"{list(completed.verdict.crossed)} beyond their bars ({_scoreline(completed.verdict)})"
        )
        if on_episode is None:
            return
        try:
            on_episode(completed)
        except Exception as exc:
            counts["unrecorded"] += 1
            metrics.observe_unrecorded_episode()
            logger.warning(f"episode at {completed.where()} was not recorded: {exc}")

    def handle_detection(msg: SourceMessage) -> None:
        counts["events"] += 1
        try:
            if msg.value is None:
                raise ValueError("detection message has no value")
            health.append(DetectionEvent.from_value(msg.value))
        except Exception as exc:
            counts["skipped_events"] += 1
            logger.warning(f"skipping undecodable detection event: {exc}")
            return
        if len(health) < health_window:
            return
        metrics.observe_prediction(prediction_health(health))
        health.clear()

    logger.info(
        f"watching {streaming.raw_frames_topic} (every {monitoring.sample_every}th frame, "
        f"windows of {monitoring.window_frames}) and {streaming.detections_topic}; bars "
        f"{_readings(reference.thresholds)} from {reference.n_scenes} training scenes"
    )
    try:
        idle_since = time.monotonic()
        while max_messages is None or counts["consumed"] < max_messages:
            msg = source()
            if msg is None:
                if idle_timeout_s is not None and time.monotonic() - idle_since > idle_timeout_s:
                    logger.info(f"idle for {idle_timeout_s:.1f}s; stopping")
                    break
                continue
            idle_since = time.monotonic()
            counts["consumed"] += 1
            if msg.topic == streaming.raw_frames_topic:
                handle_frame(msg)
            elif msg.topic == streaming.detections_topic:
                handle_detection(msg)
            else:
                logger.warning(f"ignoring a message from an unsubscribed topic {msg.topic!r}")
    except KeyboardInterrupt:
        logger.info("interrupted")
    logger.info(f"done: {counts}")
    return counts


def _readings(values: dict[str, float]) -> str:
    return " ".join(f"{name} {value:.2f}" for name, value in values.items())


def _scoreline(verdict: WindowVerdict) -> str:
    """Every score beside its own bar — the reading an operator compares, not a verdict word."""
    return " ".join(
        f"{name} {score:.2f}/{verdict.thresholds[name]:.2f}"
        for name, score in verdict.scores.items()
    )


def build_parser() -> argparse.ArgumentParser:
    return consumer_parser("Score live frames against the champion's baseline.", "messages")


def consumer_config(settings: Settings, *, offset_reset: str) -> dict:
    """A live tail that stays live: the monitor commits **nothing**.

    It produces nothing, so it needs no bookmark — and a bookmark would actively harm it.
    ``auto.offset.reset`` applies only where the group has no committed offset, so a
    committing monitor would resume mid-backlog after a restart, replay it at full speed and
    publish scores for footage from hours ago while ``drift_window_age_seconds`` reset to
    zero: stale readings wearing a live timestamp. Never committing keeps every restart a
    tail of what is happening now.
    """
    return {
        "bootstrap.servers": settings.streaming.bootstrap_servers,
        "group.id": settings.monitoring.group,
        "auto.offset.reset": offset_reset,
        "enable.auto.commit": False,
        "fetch.message.max.bytes": FRAME_MAX_MESSAGE_BYTES,  # frames exceed the 1 MiB default
    }


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    args = build_parser().parse_args(argv)

    resolved = resolve_baseline(settings)
    reference = drift_reference(resolved.scenes, settings)
    logger.info(
        f"judging against model version {resolved.model.version}'s data version "
        f"{resolved.data_run_id}"
    )
    on_episode = open_recorder(resolved)

    from confluent_kafka import Consumer
    from prometheus_client import CollectorRegistry, start_http_server

    registry = CollectorRegistry()
    metrics = MonitorMetrics(registry)
    start_http_server(settings.monitoring.metrics_port, registry=registry)
    logger.info(f"exposing metrics on :{settings.monitoring.metrics_port}")

    consumer = Consumer(consumer_config(settings, offset_reset=args.offset_reset))
    consumer.subscribe([settings.streaming.raw_frames_topic, settings.streaming.detections_topic])
    try:
        run_loop(
            settings,
            reference,
            kafka_source(consumer),
            metrics,
            on_episode=on_episode,
            max_messages=args.max_messages,
            idle_timeout_s=args.idle_timeout_s,
        )
    finally:
        consumer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
