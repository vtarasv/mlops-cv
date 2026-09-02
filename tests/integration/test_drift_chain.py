"""The synthetic shift becomes a drift episode, through a real broker.

Requires the streaming stack::

    make streaming-up
"""

from __future__ import annotations

import io
import uuid

import numpy as np
import pytest
from PIL import Image
from prometheus_client import CollectorRegistry

from mlops_cv.config import Settings, StreamingSettings, get_settings
from mlops_cv.monitoring import monitor
from mlops_cv.monitoring.drift import DriftReference, window_means
from mlops_cv.monitoring.metrics import MonitorMetrics
from mlops_cv.pipelines.profiling import BaselineScene, profile_drift_bytes
from mlops_cv.streaming.messages import pack_frame
from mlops_cv.streaming.shift import Shift

pytestmark = pytest.mark.docker

TIMEOUT = 5.0
IDLE_TIMEOUT_S = 20.0

WINDOW_FRAMES = 5  # the shipped 100 would need 500 frames per window to say the same thing
SEQUENCE = "itest-drift"


def _frame(seed: int, shade: float = 114.0, sigma: float = 40.0) -> bytes:
    """A small textured JPEG — flat colour carries no blur signal for a defocus to destroy."""
    rng = np.random.default_rng(seed)
    pixels = np.clip(rng.normal(shade, sigma, (128, 128, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _reference() -> DriftReference:
    """A training-scene cloud in miniature, built the way the published baseline is built."""
    scenes = [
        BaselineScene(
            sequence=f"train-{i}",
            n_frames=4,
            means=window_means(
                [
                    profile_drift_bytes(_frame(1000 + i * 10 + j, 100 + 4 * i, 34 + 1.5 * i))
                    for j in range(4)
                ]
            ),
        )
        for i in range(8)
    ]
    return DriftReference.from_scenes(scenes)


@pytest.fixture(scope="module")
def bootstrap() -> str:
    from confluent_kafka.admin import AdminClient

    servers = get_settings().streaming.bootstrap_servers
    try:
        AdminClient({"bootstrap.servers": servers}).list_topics(timeout=TIMEOUT)
    except Exception as exc:
        pytest.skip(f"streaming stack not reachable at {servers}: {exc}")
    return servers


@pytest.fixture
def isolated_monitor(bootstrap: str):
    """A frames topic and consumer group of this run's own, torn down after."""
    from confluent_kafka.admin import AdminClient
    from confluent_kafka.cimpl import NewTopic

    run_id = uuid.uuid4().hex[:8]
    base = Settings(
        streaming=StreamingSettings(
            bootstrap_servers=bootstrap,
            raw_frames_topic=f"itest-drift-frames-{run_id}",
            detections_topic=f"itest-drift-detections-{run_id}",
        )
    )
    settings = base.model_copy(
        update={
            "monitoring": base.monitoring.model_copy(
                update={
                    "group": f"itest-drift-{run_id}",
                    "sample_every": 1,
                    "window_frames": WINDOW_FRAMES,
                    "consecutive_windows": 2,
                }
            )
        }
    )
    admin = AdminClient({"bootstrap.servers": bootstrap})
    topic = settings.streaming.raw_frames_topic
    for future in admin.create_topics([NewTopic(topic, num_partitions=1)]).values():
        future.result(timeout=TIMEOUT)
    yield settings
    # The group goes first: deleting it needs it empty, which it is once the test closed its
    # consumer. It commits nothing, so what is left behind is a name — but a name that shows up
    # in every `rpk group list` and every lag panel until somebody reaps it.
    _delete_group(admin, settings.monitoring.group)
    for future in admin.delete_topics([topic], operation_timeout=TIMEOUT).values():
        future.result(timeout=TIMEOUT)


def _delete_group(admin, group: str) -> None:  # noqa: ANN001
    """Best effort: the broker may already have reaped an empty, never-committing group."""
    for future in admin.delete_consumer_groups([group], request_timeout=TIMEOUT).values():
        try:
            future.result(timeout=TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - teardown must not mask the test's own verdict
            print(f"could not delete consumer group {group!r}: {exc}")


def _publish(settings: Settings, payloads: list[bytes]) -> None:
    import time

    from confluent_kafka import Producer

    seeder = Producer(
        {"bootstrap.servers": settings.streaming.bootstrap_servers, "enable.idempotence": True}
    )
    for index, payload in enumerate(payloads):
        key, headers = pack_frame(SEQUENCE, frame_index=index, ts_ms=time.time_ns() // 1_000_000)
        seeder.produce(settings.streaming.raw_frames_topic, payload, key=key, headers=headers)  # type: ignore[arg-type]
    assert seeder.flush(10) == 0


def _sample(registry: CollectorRegistry, name: str, **labels: str) -> float:
    for family in registry.collect():
        for sample in family.samples:
            if sample.name == name and all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    raise AssertionError(f"the monitor exported no {name} {labels or ''}")


def test_a_dialed_shift_opens_an_episode_and_normal_traffic_re_arms_it(
    isolated_monitor: Settings,
) -> None:
    """Four drifted windows, two episodes: the rule fires once and only re-arms after clean."""
    settings = isolated_monitor
    defocus = Shift(mode="defocus", amount=4.0)
    window = WINDOW_FRAMES

    # ok, ok | drifted, drifted (episode) | ok, ok (re-arm) | drifted, drifted (episode)
    payloads: list[bytes] = []
    for phase, shift in enumerate([None, defocus, None, defocus]):
        frames = [_frame(50 + phase * 100 + i) for i in range(2 * window)]
        payloads += frames if shift is None else [shift.apply(f) for f in frames]
    _publish(settings, payloads)

    from confluent_kafka import Consumer

    registry = CollectorRegistry()
    metrics = MonitorMetrics(registry)
    episodes: list[monitor.DriftWindow] = []
    consumer = Consumer(monitor.consumer_config(settings, offset_reset="earliest"))
    consumer.subscribe([settings.streaming.raw_frames_topic])
    try:
        counts = monitor.run_loop(
            settings,
            _reference(),
            monitor.kafka_source(consumer),
            metrics,
            on_episode=episodes.append,
            max_messages=len(payloads),
            idle_timeout_s=IDLE_TIMEOUT_S,
        )
    finally:
        consumer.close()

    assert counts == {
        "consumed": 40,
        "frames": 40,
        "sampled": 40,
        "events": 0,
        "windows": 8,
        "drifted": 4,
        "episodes": 2,
        "unrecorded": 0,
        "skipped": 0,
        "skipped_events": 0,
    }

    # Two episodes out of four drifted windows is the whole claim: fire once, re-arm on clean.
    assert len(episodes) == 2
    for episode in episodes:
        assert "blur" in episode.verdict.crossed, episode.verdict.scores
        assert episode.where().startswith(f"{SEQUENCE}/"), "an episode must address its footage"

    # And the same thing read the way an operator reads it — off the monitor's own collectors.
    assert _sample(registry, "drift_episodes_total") == 2
    assert _sample(registry, "drift_windows_total", verdict="drifted") == 4
    assert _sample(registry, "drift_windows_total", verdict="ok") == 4
    assert _sample(registry, "drift_score", statistic="blur") > _sample(
        registry, "drift_threshold", statistic="blur"
    ), "the last window was shifted, so the live score must be sitting above its bar"
