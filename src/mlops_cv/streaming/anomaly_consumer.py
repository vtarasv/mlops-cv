"""Anomaly consumer: Detection events in, Alerts out."""

from __future__ import annotations

import argparse
import logging
import time

from mlops_cv.config import Settings, get_settings
from mlops_cv.streaming import consumer_parser
from mlops_cv.streaming.anomaly import EpisodeRule
from mlops_cv.streaming.messages import DetectionEvent

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    return consumer_parser("Watch Detection events; publish Alerts.", "events")


def run_loop(
    settings: Settings,
    *,
    offset_reset: str = "latest",
    max_messages: int | None = None,
    idle_timeout_s: float | None = None,
) -> dict[str, int]:
    """Consume Detection events, publish Alerts; return counters when bounded."""
    from confluent_kafka import Consumer, Producer

    streaming = settings.streaming
    rule = EpisodeRule(
        cls=streaming.anomaly_class,
        window_s=streaming.anomaly_window_s,
        threshold=streaming.anomaly_threshold,
    )
    consumer = Consumer(
        {
            "bootstrap.servers": streaming.bootstrap_servers,
            "group.id": streaming.anomaly_group,
            "auto.offset.reset": offset_reset,
            "enable.auto.commit": True,
        }
    )
    producer = Producer(
        {"bootstrap.servers": streaming.bootstrap_servers, "enable.idempotence": True}
    )
    consumer.subscribe([streaming.detections_topic])
    logger.info(
        f"watching {streaming.detections_topic}: "
        f"windowed mean {rule.cls}-count > {rule.threshold} over {rule.window_s}s"
    )

    counts = {"consumed": 0, "alerts": 0, "skipped": 0}
    try:
        idle_since = time.monotonic()
        while max_messages is None or counts["consumed"] < max_messages:
            msg = consumer.poll(0.2)
            producer.poll(0)
            if msg is None:
                if idle_timeout_s is not None and time.monotonic() - idle_since > idle_timeout_s:
                    logger.info(f"idle for {idle_timeout_s:.1f}s; stopping")
                    break
                continue
            if msg.error():
                logger.warning(f"consumer error: {msg.error()}")
                continue
            idle_since = time.monotonic()
            counts["consumed"] += 1

            try:
                value = msg.value()
                if value is None:
                    raise ValueError("detection message has no value")
                event = DetectionEvent.from_value(value)
            except Exception as exc:
                counts["skipped"] += 1
                logger.warning(f"skipping undecodable detection event: {exc}")
                continue

            alert = rule.observe(event)
            if alert is not None:
                counts["alerts"] += 1
                logger.info(
                    f"ALERT: {alert.rule} {alert.cls}-count "
                    f"{alert.observed:.1f} > {alert.threshold:.1f} "
                    f"at {alert.sequence}/{alert.frame_index}"
                )
                producer.produce(streaming.alerts_topic, alert.to_value(), key=alert.kafka_key())
    except KeyboardInterrupt:
        logger.info("interrupted; flushing")
    finally:
        producer.flush(10)
        consumer.close()
    logger.info(f"done: {counts}")
    return counts


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    args = build_parser().parse_args(argv)
    run_loop(
        settings,
        offset_reset=args.offset_reset,
        max_messages=args.max_messages,
        idle_timeout_s=args.idle_timeout_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
