"""GPU inference consumer: Frame messages in, Detection events out.

Serves the champion's **published ONNX graph** through the serving detector — the same
artifact, pre/post math, and class names (read from the graph's own metadata) that answer
``/predict``, so the bus and HTTP cannot drift apart. Resolution happens once at startup
(restart to pick up a promotion; every event self-identifies its model, so a changeover is
visible in the stream), and both empty-registry states exit 2 with the command that fixes
them, exactly like the service. The Kafka loop is factored around an injectable ``infer``
callable.

Delivery is **at-least-once**: offsets are *stored* only in the delivery callback of the
corresponding Detection event (i.e. after the broker confirmed the produce) and committed
by librdkafka's timer; a crash replays the uncommitted tail, and duplicate events are
idempotently addressed by ``(sequence, frame_index)``. Undecodable frames are logged,
counted, and skipped-with-stored-offset — a poison message must not wedge the partition.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from mlops_cv.config import Settings, get_settings
from mlops_cv.streaming.messages import (
    FRAME_MAX_MESSAGE_BYTES,
    Box,
    DetectionEvent,
    ModelInfo,
    parse_frame,
)

if TYPE_CHECKING:
    from confluent_kafka import Message

    from mlops_cv.serving.runtime import Detector

logger = logging.getLogger(__name__)

# The frame's decoded answer: a list of boxes (the loop owns the timing around the call).
InferenceFn = Callable[[bytes], list[Box]]

# Producer-queue watermarks for pause/resume back-pressure: pause frame intake when this
# many events await delivery, resume once the queue drains below the low mark.
PAUSE_AT_PENDING = 64
RESUME_AT_PENDING = 8

LOG_EVERY = 100


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run streaming inference on Frame messages.")
    p.add_argument(
        "--offset-reset",
        choices=("latest", "earliest"),
        default="latest",
        help="where a NEW consumer group starts: latest = live tail, earliest = replay",
    )
    p.add_argument(
        "--max-messages", type=int, default=None, help="stop after this many frames (tests/smokes)"
    )
    p.add_argument(
        "--idle-timeout-s",
        type=float,
        default=None,
        help="stop after this long with no frames (tests/smokes)",
    )
    return p


def consumer_config(settings: Settings, *, offset_reset: str) -> dict:
    """The at-least-once consumer configuration.

    ``enable.auto.commit`` stays on but commits only what the loop *stored* — never the
    poll position (``enable.auto.offset.store=false``) — so the librdkafka timer plus the
    final commit in ``close()`` persist exactly the delivery-confirmed offsets.
    """
    return {
        "bootstrap.servers": settings.streaming.bootstrap_servers,
        "group.id": settings.streaming.inference_group,
        "auto.offset.reset": offset_reset,
        "enable.auto.commit": True,
        "auto.commit.interval.ms": int(settings.streaming.commit_interval_s * 1000),
        "enable.auto.offset.store": False,
        "fetch.message.max.bytes": FRAME_MAX_MESSAGE_BYTES,  # frames may exceed the 1 MiB default
    }


def producer_config(settings: Settings) -> dict:
    return {
        "bootstrap.servers": settings.streaming.bootstrap_servers,
        "enable.idempotence": True,
    }


def run_loop(
    settings: Settings,
    infer: InferenceFn,
    model: ModelInfo,
    *,
    offset_reset: str = "latest",
    max_messages: int | None = None,
    idle_timeout_s: float | None = None,
) -> dict[str, int]:
    """Consume Frame messages, publish Detection events; return counters when bounded.

    Runs forever unless ``max_messages`` / ``idle_timeout_s`` bounds it (tests, smokes).
    """
    from confluent_kafka import Consumer, Producer, TopicPartition

    consumer = Consumer(consumer_config(settings, offset_reset=offset_reset))
    producer = Producer(producer_config(settings))
    consumer.subscribe([settings.streaming.raw_frames_topic])

    counts = {"consumed": 0, "events": 0, "skipped": 0, "delivery_failures": 0}
    paused = False
    stored_next: dict[tuple[str, int], int] = {}  # (topic, partition) -> next offset stored

    def store_frame_offset(source: Message) -> None:
        """Advance the partition's stored offset — monotonically.

        Skipped frames store synchronously while earlier frames' delivery callbacks are
        still in flight; without the max-guard a late callback would move the stored
        offset *backwards* and re-deliver already-acknowledged frames on restart.
        """
        topic, partition, offset = source.topic(), source.partition(), source.offset()
        if topic is None or partition is None or offset is None:
            # Only unset on producer-side Messages; a consumed frame always carries all three.
            logger.warning(f"frame without an address ({topic}/{partition}@{offset})")
            return
        tp = (topic, partition)
        nxt = offset + 1
        if nxt > stored_next.get(tp, -1):
            stored_next[tp] = nxt
            consumer.store_offsets(offsets=[TopicPartition(topic, partition, nxt)])

    def on_delivery(err, event_msg, source: Message) -> None:
        """Store the *frame's* offset only once its event is broker-confirmed."""
        if err is not None:
            # Fail fast: with an idempotent producer a delivery failure is not a blip,
            # and continuing could let a later stored offset commit past this frame.
            counts["delivery_failures"] += 1
            logger.error(f"event delivery failed for {event_msg.key()}: {err}")
            return
        store_frame_offset(source)

    try:
        idle_since = time.monotonic()
        while max_messages is None or counts["consumed"] < max_messages:
            if counts["delivery_failures"]:
                logger.error("stopping after a detection-event delivery failure")
                break
            # Back-pressure: stop taking frames while too many events await delivery.
            if not paused and len(producer) >= PAUSE_AT_PENDING:
                consumer.pause(consumer.assignment())
                paused = True
                logger.info(f"paused intake ({len(producer)} events pending delivery)")
            if paused and len(producer) <= RESUME_AT_PENDING:
                consumer.resume(consumer.assignment())
                paused = False

            msg = consumer.poll(0.2)
            producer.poll(0)  # serve delivery callbacks (they store offsets)
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
                ref = parse_frame(msg.key(), msg.headers())
                payload = msg.value()
                if payload is None:  # a tombstone carries no JPEG — treat it as poison
                    raise ValueError("frame message has no value")
                started = time.perf_counter()
                boxes = infer(payload)
                latency_ms = (time.perf_counter() - started) * 1000.0
            except Exception as exc:
                # Poison frame: record it, keep the partition moving (offset still stored).
                counts["skipped"] += 1
                logger.warning(f"skipping undecodable frame {msg.key()}: {exc}")
                store_frame_offset(msg)
                continue

            event = DetectionEvent(
                sequence=ref.sequence,
                frame_index=ref.frame_index,
                ts_frame_ms=ref.ts_ms,
                ts_infer_ms=time.time_ns() // 1_000_000,
                latency_ms=latency_ms,
                model=model,
                boxes=boxes,
            )
            producer.produce(
                settings.streaming.detections_topic,
                event.to_value(),
                key=event.kafka_key(),
                on_delivery=lambda err, event_msg, source=msg: on_delivery(err, event_msg, source),
            )
            counts["events"] += 1
            if counts["events"] % LOG_EVERY == 0:
                logger.info(f"published {counts['events']} detection events")
    except KeyboardInterrupt:
        logger.info("interrupted; flushing")
    finally:
        producer.flush(10)
        consumer.close()  # final auto-commit of stored offsets
    logger.info(f"done: {counts}")
    return counts


def detector_inference(detector: Detector, conf_threshold: float) -> InferenceFn:
    """Production wiring of the injectable seam: the serving detector on JPEG bytes."""

    def infer(jpeg: bytes) -> list[Box]:
        return detector.detect(jpeg, conf_threshold)

    return infer


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    args = build_parser().parse_args(argv)

    from mlops_cv.serving.errors import StartupError
    from mlops_cv.serving.resolve import champion_detector
    from mlops_cv.tracking import client

    client.configure(settings)
    try:
        detector = champion_detector(settings)
    except StartupError as exc:
        logger.error(str(exc))
        return 2

    run_loop(
        settings,
        detector_inference(detector, settings.serving.conf_threshold),
        detector.model,
        offset_reset=args.offset_reset,
        max_messages=args.max_messages,
        idle_timeout_s=args.idle_timeout_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
