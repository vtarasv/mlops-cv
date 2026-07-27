"""Broker round-trip tests for the streaming processes against the live stack.

Requires the stack to be up::

    make streaming-up
"""

from __future__ import annotations

import io
import time
import uuid
from pathlib import Path

import pytest
from PIL import Image

from mlops_cv.config import Settings, StreamingSettings, get_settings
from mlops_cv.streaming import anomaly_consumer, inference_consumer, producer
from mlops_cv.streaming.messages import (
    Alert,
    Box,
    DetectionEvent,
    ModelInfo,
    pack_frame,
    parse_frame,
)

pytestmark = pytest.mark.docker

TIMEOUT = 5.0
CONSUME_DEADLINE = 20.0


def _require_broker() -> str:
    from confluent_kafka.admin import AdminClient

    servers = get_settings().streaming.bootstrap_servers
    try:
        AdminClient({"bootstrap.servers": servers}).list_topics(timeout=TIMEOUT)
    except Exception as exc:
        pytest.skip(f"streaming stack not reachable at {servers}: {exc}")
    return servers


@pytest.fixture(scope="module")
def bootstrap() -> str:
    return _require_broker()


@pytest.fixture
def demo_store(tmp_path: Path) -> tuple[Path, list[str]]:
    """A tiny synthetic demo store with unique sequence names (the shared topic filter key)."""
    run_id = uuid.uuid4().hex[:8]
    sequences = [f"itest-{run_id}-a", f"itest-{run_id}-b"]
    for shade, sequence in enumerate(sequences):
        seq_dir = tmp_path / "demo" / "images" / sequence
        seq_dir.mkdir(parents=True)
        for index in (1, 21, 41):
            Image.new("RGB", (16, 16), (shade * 100, 10, 10)).save(seq_dir / f"{index:07d}.jpg")
    return tmp_path, sequences


def _delete_topics(admin, names: list[str]) -> None:  # noqa: ANN001
    """Delete and *wait*: delete_topics is async, so fire-and-forget leaks a topic per run."""
    for future in admin.delete_topics(names, operation_timeout=TIMEOUT).values():
        future.result(timeout=TIMEOUT)


def _consume(bootstrap: str, topic: str, *, key_prefix: bytes, expect: int) -> list:
    """Read the topic from the start; collect ``expect`` messages whose key matches."""
    import time

    from confluent_kafka import Consumer

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": f"itest-{uuid.uuid4().hex[:8]}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,  # a probe, not a pipeline member: never move offsets
        }
    )
    consumer.subscribe([topic])
    collected: list = []
    deadline = time.monotonic() + CONSUME_DEADLINE
    try:
        while len(collected) < expect and time.monotonic() < deadline:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            if msg.key() and msg.key().startswith(key_prefix):  # type: ignore[union-attr]
                collected.append(msg)
    finally:
        consumer.close()
    return collected


@pytest.fixture
def isolated_streaming(bootstrap: str):
    """Unique frames/detections topics + consumer group, torn down after the test.

    The inference-consumer chain needs deterministic contents, so it must not share the
    stack's real topics with other tests' leftovers.
    """
    from confluent_kafka.admin import AdminClient
    from confluent_kafka.cimpl import NewTopic

    run_id = uuid.uuid4().hex[:8]
    streaming = StreamingSettings(
        bootstrap_servers=bootstrap,
        raw_frames_topic=f"itest-frames-{run_id}",
        detections_topic=f"itest-detections-{run_id}",
        inference_group=f"itest-inference-{run_id}",
        commit_interval_s=1.0,
    )
    admin = AdminClient({"bootstrap.servers": bootstrap})
    futures = admin.create_topics(
        [
            NewTopic(streaming.raw_frames_topic, num_partitions=3),
            NewTopic(streaming.detections_topic, num_partitions=3),
        ]
    )
    for future in futures.values():
        future.result(timeout=TIMEOUT)
    yield Settings(streaming=streaming)
    _delete_topics(admin, [streaming.raw_frames_topic, streaming.detections_topic])


def _seed_frames(bootstrap: str, topic: str, frames: list[tuple[str, int, bytes]]) -> None:
    from confluent_kafka import Producer

    seeder = Producer({"bootstrap.servers": bootstrap, "enable.idempotence": True})
    for sequence, index, payload in frames:
        key, headers = pack_frame(sequence, frame_index=index, ts_ms=time.time_ns() // 1_000_000)
        seeder.produce(topic, payload, key=key, headers=headers)  # type: ignore[arg-type]
    assert seeder.flush(10) == 0


def _jpeg(shade: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (shade, 20, 20)).save(buf, format="JPEG")
    return buf.getvalue()


def test_inference_loop_stub_model_chain(isolated_streaming: Settings) -> None:
    """Frames in -> Detection events out, offsets committed only for delivered events,
    poison frames skipped without wedging the partition."""
    from confluent_kafka import Consumer, TopicPartition

    settings = isolated_streaming
    streaming = settings.streaming
    frames = [("seqA", i, _jpeg(i * 20)) for i in (1, 21, 41)] + [
        ("seqB", i, _jpeg(i * 10)) for i in (1, 21)
    ]
    frames.append(("seqB", 41, b"not a jpeg"))  # poison: must be skipped, not wedge
    _seed_frames(streaming.bootstrap_servers, streaming.raw_frames_topic, frames)

    def stub_infer(jpeg: bytes) -> list[Box]:
        with Image.open(io.BytesIO(jpeg)) as im:  # raises on the poison frame
            im.load()
        return [Box(cls="person", conf=0.9, xywhn=(0.5, 0.5, 0.1, 0.1))]

    counts = inference_consumer.run_loop(
        settings,
        stub_infer,
        ModelInfo(name="stub", version="0"),
        offset_reset="earliest",
        max_messages=6,
        idle_timeout_s=15.0,
    )
    assert counts == {"consumed": 6, "events": 5, "skipped": 1, "delivery_failures": 0}

    # Every good frame produced a well-formed, co-partitioned Detection event.
    messages = _consume(
        streaming.bootstrap_servers, streaming.detections_topic, key_prefix=b"seq", expect=5
    )
    assert len(messages) == 5
    events = [DetectionEvent.from_value(m.value()) for m in messages]
    for msg, event in zip(messages, events, strict=True):
        assert msg.key() == event.kafka_key()  # co-partitioned with the frames
        assert event.model == ModelInfo(name="stub", version="0")
        assert event.latency_ms >= 0
        assert event.boxes[0].cls == "person"
    assert {(e.sequence, e.frame_index) for e in events} == {
        ("seqA", 1),
        ("seqA", 21),
        ("seqA", 41),
        ("seqB", 1),
        ("seqB", 21),
    }

    # At-least-once bookkeeping: every consumed frame's offset (incl. the poison skip)
    # was stored and committed by the loop's close().
    probe = Consumer(
        {
            "bootstrap.servers": streaming.bootstrap_servers,
            "group.id": streaming.inference_group,
            "enable.auto.commit": False,
        }
    )
    committed = probe.committed(
        [TopicPartition(streaming.raw_frames_topic, p) for p in range(3)], timeout=TIMEOUT
    )
    probe.close()
    assert sum(tp.offset for tp in committed if tp.offset >= 0) == 6


def test_anomaly_consumer_emits_one_alert_episode(bootstrap: str) -> None:
    """Synthetic crowding in -> exactly one Alert on the alerts topic (episode semantics)."""
    from confluent_kafka import Producer
    from confluent_kafka.admin import AdminClient
    from confluent_kafka.cimpl import NewTopic

    run_id = uuid.uuid4().hex[:8]
    streaming = StreamingSettings(
        bootstrap_servers=bootstrap,
        detections_topic=f"itest-detections-{run_id}",
        alerts_topic=f"itest-alerts-{run_id}",
        anomaly_group=f"itest-anomaly-{run_id}",
        anomaly_class="person",
        anomaly_window_s=5.0,
        anomaly_threshold=2.5,
    )
    admin = AdminClient({"bootstrap.servers": bootstrap})
    futures = admin.create_topics(
        [
            NewTopic(streaming.detections_topic, num_partitions=3),
            NewTopic(streaming.alerts_topic, num_partitions=1),
        ]
    )
    for future in futures.values():
        future.result(timeout=TIMEOUT)
    try:
        # Counts 1,1 (calm) then 5,5,5 (crowding: crossing fires ONCE) then 1 (clears).
        seeder = Producer({"bootstrap.servers": bootstrap, "enable.idempotence": True})
        for i, n_person in enumerate([1, 1, 5, 5, 5, 1]):
            event = DetectionEvent(
                sequence="crowd",
                frame_index=i,
                ts_frame_ms=1_000 * (i + 1),
                ts_infer_ms=1_000 * (i + 1) + 20,
                latency_ms=20.0,
                model=ModelInfo(name="stub", version="0"),
                boxes=[Box(cls="person", conf=0.9, xywhn=(0.5, 0.5, 0.1, 0.1))] * n_person,
            )
            seeder.produce(streaming.detections_topic, event.to_value(), key=event.kafka_key())
        assert seeder.flush(10) == 0

        counts = anomaly_consumer.run_loop(
            Settings(streaming=streaming),
            offset_reset="earliest",
            max_messages=6,
            idle_timeout_s=15.0,
        )
        assert counts == {"consumed": 6, "alerts": 1, "skipped": 0}

        messages = _consume(bootstrap, streaming.alerts_topic, key_prefix=b"crowd", expect=1)
        assert len(messages) == 1, "exactly one episode despite three above-threshold frames"
        alert = Alert.from_value(messages[0].value())
        assert alert.rule == "windowed-count"
        assert alert.cls == "person"
        assert alert.observed > alert.threshold == 2.5
        assert alert.sequence == "crowd"
    finally:
        _delete_topics(admin, [streaming.detections_topic, streaming.alerts_topic])


def test_producer_roundtrip_bytes_key_headers_survive(
    bootstrap: str, demo_store: tuple[Path, list[str]]
) -> None:
    root, sequences = demo_store
    assert producer.main(["--subset-dir", str(root), "--loops", "1", "--fps", "0"]) == 0

    prefix = sequences[0].rsplit("-", 1)[0].encode()  # itest-<run_id>
    messages = _consume(
        bootstrap, get_settings().streaming.raw_frames_topic, key_prefix=prefix, expect=6
    )
    assert len(messages) == 6, "expected all 6 published frames back"

    by_ref = {}
    for msg in messages:
        ref = parse_frame(msg.key(), msg.headers())
        assert ref.ts_ms > 0
        by_ref[(ref.sequence, ref.frame_index)] = msg.value()
    for sequence in sequences:
        for index in (1, 21, 41):
            original = (root / "demo" / "images" / sequence / f"{index:07d}.jpg").read_bytes()
            assert by_ref[(sequence, index)] == original, "bytes must survive verbatim"
