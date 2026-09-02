"""The monitor's run loop, driven through a faked bus and a faked episode sink."""

from __future__ import annotations

import io
from collections.abc import Callable

import pytest
from PIL import Image
from prometheus_client import CollectorRegistry

from mlops_cv.config import Settings
from mlops_cv.monitoring import monitor
from mlops_cv.monitoring.drift import DriftReference, WindowVerdict
from mlops_cv.monitoring.metrics import MonitorMetrics
from mlops_cv.monitoring.monitor import DriftWindow, SourceMessage, prediction_health, run_loop
from mlops_cv.pipelines.profiling import BaselineScene, profile_drift_bytes
from mlops_cv.startup import StartupError
from mlops_cv.streaming.messages import Box, DetectionEvent, FrameRef, ModelInfo, pack_frame

FRAMES_TOPIC = Settings().streaming.raw_frames_topic
DETECTIONS_TOPIC = Settings().streaming.detections_topic
MODEL = ModelInfo(name="aerial-object-detector", version="8")


# --- doubles ---


def _frame_bytes(level: int) -> bytes:
    """A small PNG at a given brightness, with edges so blur/contrast are non-degenerate."""
    im = Image.new("L", (32, 32))
    pixels = im.load()
    assert pixels is not None
    for y in range(32):
        for x in range(32):
            pixels[x, y] = level if (x // 4 + y // 4) % 2 else max(level - 40, 0)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


NORMAL = _frame_bytes(160)
DARK = _frame_bytes(40)


def _frame(
    payload: bytes | None = NORMAL, *, sequence: str = "seqA", index: int = 0
) -> SourceMessage:
    key, headers = pack_frame(sequence, frame_index=index, ts_ms=1_700_000_000_000 + index)
    return SourceMessage(topic=FRAMES_TOPIC, key=key, value=payload, headers=headers)


def _detection(*, boxes: list[Box], index: int = 0) -> SourceMessage:
    event = DetectionEvent(
        sequence="seqA",
        frame_index=index,
        ts_frame_ms=1_700_000_000_000,
        ts_infer_ms=1_700_000_000_010,
        latency_ms=4.0,
        model=MODEL,
        boxes=boxes,
    )
    return SourceMessage(topic=DETECTIONS_TOPIC, key=event.kafka_key(), value=event.to_value())


def _source(messages: list[SourceMessage]) -> Callable[[], SourceMessage | None]:
    """A bus that hands over a script and then goes quiet (the loop's idle timeout ends it)."""
    queue = list(messages)
    return lambda: queue.pop(0) if queue else None


def _settings(**monitoring) -> Settings:
    settings = Settings()
    return settings.model_copy(
        update={"monitoring": settings.monitoring.model_copy(update=monitoring)}
    )


@pytest.fixture
def reference() -> DriftReference:
    """A cloud centred on what NORMAL actually measures, ±10% — so NORMAL scores ~0."""
    means = profile_drift_bytes(NORMAL)
    scenes = [
        BaselineScene(
            sequence=f"seq{i}", n_frames=10, means={k: v * factor for k, v in means.items()}
        )
        for i, factor in enumerate((0.9, 1.0, 1.1))
    ]
    return DriftReference.from_scenes(scenes)


@pytest.fixture
def metrics() -> MonitorMetrics:
    """Collectors on their own registry — never the process-global one."""
    return MonitorMetrics(CollectorRegistry())


def _drive(settings: Settings, reference, metrics, messages, **kwargs) -> dict[str, int]:
    return run_loop(settings, reference, _source(messages), metrics, idle_timeout_s=0.0, **kwargs)


# --- the reference itself ---


def test_the_normal_frame_is_not_drifted_against_its_own_cloud(reference: DriftReference) -> None:
    """Guards the fixture: if NORMAL scored drifted, every loop test below would be vacuous."""
    assert not reference.judge(profile_drift_bytes(NORMAL)).drifted
    assert reference.judge(profile_drift_bytes(DARK)).drifted


# --- sampling and windowing ---


def test_only_every_kth_frame_is_profiled(reference, metrics) -> None:
    """Sampling is not optional: profiling every frame cannot keep up with the producer."""
    settings = _settings(sample_every=5, window_frames=100)
    counts = _drive(settings, reference, metrics, [_frame(index=i) for i in range(10)])

    assert counts["frames"] == 10
    assert counts["sampled"] == 2


def test_windows_are_tumbling_not_sliding(reference, metrics) -> None:
    """Four sampled frames at window 2 make two windows, not three: no frame is judged twice."""
    settings = _settings(sample_every=1, window_frames=2)
    frames = [_frame(DARK, index=0), _frame(DARK, index=1), _frame(index=2), _frame(index=3)]

    counts = _drive(settings, reference, metrics, frames)

    assert counts["windows"] == 2
    assert counts["drifted"] == 1  # a sliding window would straddle the boundary and find two


def test_an_incomplete_window_is_never_judged(reference, metrics) -> None:
    settings = _settings(sample_every=1, window_frames=3)
    counts = _drive(settings, reference, metrics, [_frame(index=i) for i in range(2)])

    assert counts["sampled"] == 2
    assert counts["windows"] == 0


def test_a_frame_that_fails_to_decode_is_skipped_and_counted(reference, metrics) -> None:
    """A poison frame must not wedge the monitor — the consumers' rule, applied here."""
    settings = _settings(sample_every=1, window_frames=2)
    frames = [_frame(index=0), _frame(b"not an image", index=1), _frame(index=2)]

    counts = _drive(settings, reference, metrics, frames)

    assert counts["skipped"] == 1
    assert counts["windows"] == 1  # the two decodable frames still completed a window


def test_a_frame_without_a_value_is_skipped_too(reference, metrics) -> None:
    """A well-addressed tombstone: the headers parse, so only the value guard can catch it."""
    settings = _settings(sample_every=1, window_frames=1)

    counts = _drive(settings, reference, metrics, [_frame(None, index=0)])

    assert counts["skipped"] == 1
    assert counts["windows"] == 0


def test_a_frame_with_no_address_is_skipped_too(reference, metrics) -> None:
    """The other half: a frame whose headers violate the wire contract."""
    settings = _settings(sample_every=1, window_frames=1)
    unaddressed = SourceMessage(topic=FRAMES_TOPIC, key=b"seqA", value=NORMAL, headers=[])

    assert _drive(settings, reference, metrics, [unaddressed])["skipped"] == 1


# --- where a window's footage is ---


def _window(*sequences: tuple[str, int]) -> DriftWindow:
    verdict = WindowVerdict(scores={}, thresholds={}, crossed=())
    frames = [
        FrameRef(sequence=sequence, frame_index=index, ts_ms=1) for sequence, index in sequences
    ]
    return DriftWindow(verdict=verdict, frames=tuple(frames))


def test_each_clip_gets_its_own_span() -> None:
    """A window may cover several clips; one min..max across them addresses footage it never saw."""
    window = _window(("seqA", 90), ("seqA", 95), ("seqB", 0))

    assert window.spans() == [
        {"sequence": "seqA", "first": 90, "last": 95, "n_frames": 2},
        {"sequence": "seqB", "first": 0, "last": 0, "n_frames": 1},
    ]


def test_interleaved_partitions_do_not_fragment_the_address() -> None:
    """Measured live: clips alternate every few frames, so a split per switch would describe the
    partition schedule rather than the footage (8 ranges for a 20-frame window)."""
    window = _window(*[(seq, i) for i in range(4) for seq in ("seqA", "seqB")])

    assert window.spans() == [
        {"sequence": "seqA", "first": 0, "last": 3, "n_frames": 4},
        {"sequence": "seqB", "first": 0, "last": 3, "n_frames": 4},
    ]


def test_where_never_ranges_between_two_different_clips() -> None:
    """The human-readable address in every window log line and in the stored episode."""
    assert _window(("seqA", 9), ("seqB", 28), ("seqA", 16)).where() == "seqA/9-16, seqB/28-28"
    assert _window(("seqA", 0), ("seqA", 10)).where() == "seqA/0-10"


# --- the episode sink ---


def test_an_episode_reaches_the_sink_exactly_once(reference, metrics) -> None:
    """Two consecutive drifted windows open it; it stays silent while the shift holds."""
    settings = _settings(sample_every=1, window_frames=1)
    opened: list[DriftWindow] = []

    counts = _drive(
        settings,
        reference,
        metrics,
        [_frame(DARK, index=i) for i in range(4)],
        on_episode=opened.append,
    )

    assert counts["windows"] == 4
    assert counts["episodes"] == 1
    assert len(opened) == 1


def test_a_single_drifted_window_is_not_news(reference, metrics) -> None:
    settings = _settings(sample_every=1, window_frames=1)
    frames = [_frame(DARK, index=0), _frame(index=1), _frame(DARK, index=2)]

    assert _drive(settings, reference, metrics, frames)["episodes"] == 0


def test_a_clean_window_re_arms_the_episode(reference, metrics) -> None:
    settings = _settings(sample_every=1, window_frames=1)
    pattern = [DARK, DARK, NORMAL, DARK, DARK]
    frames = [_frame(payload, index=i) for i, payload in enumerate(pattern)]

    assert _drive(settings, reference, metrics, frames)["episodes"] == 2


def test_the_episode_carries_its_scores_and_the_frames_it_saw(reference, metrics) -> None:
    """What a reader needs later: which statistics crossed, how far, and where to look."""
    settings = _settings(sample_every=1, window_frames=2)
    opened: list[DriftWindow] = []
    frames = [_frame(DARK, sequence="uav0000161", index=i) for i in range(4)]

    _drive(settings, reference, metrics, frames, on_episode=opened.append)

    (episode,) = opened
    assert "brightness" in episode.verdict.crossed
    assert episode.verdict.scores["brightness"] > episode.verdict.thresholds["brightness"]
    assert [ref.frame_index for ref in episode.frames] == [2, 3]  # the window that opened it
    assert episode.frames[0].sequence == "uav0000161"


def test_a_sink_that_fails_does_not_take_the_monitor_down(reference, metrics) -> None:
    """Recording is best effort: an episode is the worst moment to lose the live signal too."""
    settings = _settings(sample_every=1, window_frames=1)

    def refuse(window: DriftWindow) -> None:
        raise RuntimeError("tracking server unreachable")

    counts = _drive(
        settings, reference, metrics, [_frame(DARK, index=i) for i in range(4)], on_episode=refuse
    )

    assert counts["episodes"] == 1
    assert counts["unrecorded"] == 1
    assert counts["windows"] == 4  # kept scoring after the failure
    assert metrics.registry.get_sample_value("drift_episodes_unrecorded_total") == 1.0


def test_a_ledger_that_never_opened_refuses_rather_than_vanishing(monkeypatch) -> None:
    """The total-loss case must not be the one failure the loss counter cannot see."""
    from mlops_cv.monitoring import evidence

    def refuse(resolved, **kwargs):  # noqa: ANN001, ANN202
        raise RuntimeError("tracking server unreachable")

    monkeypatch.setattr(evidence, "open_record", refuse)
    sink = monitor.open_recorder(object())  # type: ignore[arg-type]

    assert sink is not None
    verdict = WindowVerdict(scores={}, thresholds={}, crossed=("brightness",))
    with pytest.raises(RuntimeError, match="tracking server unreachable"):
        sink(DriftWindow(verdict=verdict, frames=()))


def test_a_recorded_episode_counts_no_loss(reference, metrics) -> None:
    settings = _settings(sample_every=1, window_frames=1)
    counts = _drive(
        settings,
        reference,
        metrics,
        [_frame(DARK, index=i) for i in range(4)],
        on_episode=lambda window: None,
    )

    assert counts["unrecorded"] == 0
    assert metrics.registry.get_sample_value("drift_episodes_unrecorded_total") == 0.0


def test_the_consecutive_setting_drives_the_rule(reference, metrics) -> None:
    settings = _settings(sample_every=1, window_frames=1, consecutive_windows=3)
    frames = [_frame(DARK, index=i) for i in range(2)]

    assert _drive(settings, reference, metrics, frames)["episodes"] == 0


# --- prediction health ---


def test_prediction_health_aggregates_a_window_of_events() -> None:
    """A pure aggregate: per-frame detection count, mean confidence, class mix."""
    events = [
        DetectionEvent(
            sequence="seqA",
            frame_index=0,
            ts_frame_ms=0,
            ts_infer_ms=1,
            latency_ms=1.0,
            model=MODEL,
            boxes=[
                Box(cls="person", conf=0.8, xywhn=(0.1, 0.1, 0.1, 0.1)),
                Box(cls="vehicle", conf=0.6, xywhn=(0.2, 0.2, 0.1, 0.1)),
            ],
        ),
        DetectionEvent(
            sequence="seqA",
            frame_index=1,
            ts_frame_ms=0,
            ts_infer_ms=1,
            latency_ms=1.0,
            model=MODEL,
            boxes=[Box(cls="person", conf=1.0, xywhn=(0.1, 0.1, 0.1, 0.1))],
        ),
    ]

    health = prediction_health(events)

    assert health.detections_per_frame == pytest.approx(1.5)
    assert health.mean_confidence == pytest.approx(0.8)
    assert health.class_shares == pytest.approx({"person": 2 / 3, "vehicle": 1 / 3})


def test_prediction_health_survives_a_window_that_detected_nothing() -> None:
    """An empty frame is a reading, not an error — division by zero must not kill the loop."""
    empty = DetectionEvent(
        sequence="seqA",
        frame_index=0,
        ts_frame_ms=0,
        ts_infer_ms=1,
        latency_ms=1.0,
        model=MODEL,
        boxes=[],
    )

    health = prediction_health([empty])

    assert health.detections_per_frame == 0.0
    assert health.mean_confidence == 0.0
    assert health.class_shares == {}


def test_the_monitor_never_commits_its_position() -> None:
    """A bookmark would resume a restarted monitor mid-backlog and publish stale scores under
    a fresh window age; the monitor produces nothing, so it needs none."""
    config = monitor.consumer_config(Settings(), offset_reset="latest")

    assert config["enable.auto.commit"] is False
    assert config["auto.offset.reset"] == "latest"
    assert config["group.id"] == Settings().monitoring.group


def test_detection_events_are_charted_and_never_judged(reference, metrics) -> None:
    """Prediction health carries no threshold: it was measured not to separate drift here."""
    settings = _settings(sample_every=1, window_frames=2)
    box = Box(cls="person", conf=0.5, xywhn=(0.1, 0.1, 0.1, 0.1))
    events = [_detection(boxes=[box, box], index=i) for i in range(2)]

    counts = _drive(settings, reference, metrics, events)

    assert counts["events"] == 2
    assert counts["episodes"] == 0
    assert counts["windows"] == 0  # detection events never complete a *drift* window
    assert metrics.registry.get_sample_value("prediction_detections_per_frame") == 2.0
    assert metrics.registry.get_sample_value("prediction_mean_confidence") == 0.5
    assert metrics.registry.get_sample_value("prediction_class_share", {"cls": "person"}) == 1.0


def test_an_undecodable_detection_event_is_skipped(reference, metrics) -> None:
    """Counted apart from unprofilable frames, so the skip counter can be attributed."""
    settings = _settings(sample_every=1, window_frames=1)
    poison = SourceMessage(topic=DETECTIONS_TOPIC, key=b"seqA", value=b"{not json")

    counts = _drive(settings, reference, metrics, [poison])

    assert counts["skipped_events"] == 1
    assert counts["skipped"] == 0  # that gauge is the *frame* skip counter


def test_prediction_health_windows_the_same_stretch_of_stream_as_drift(reference, metrics) -> None:
    """The two panels sit side by side, so they must span the same interval — not 5x apart."""
    settings = _settings(sample_every=3, window_frames=2)  # 6 frames of stream
    box = Box(cls="person", conf=0.5, xywhn=(0.1, 0.1, 0.1, 0.1))
    events = [_detection(boxes=[box, box], index=i) for i in range(6)]

    _drive(settings, reference, metrics, events[:5])
    assert metrics.registry.get_sample_value("prediction_detections_per_frame") == 0.0  # unset

    _drive(settings, reference, metrics, events)
    assert metrics.registry.get_sample_value("prediction_detections_per_frame") == 2.0


def test_the_prediction_window_age_reports_a_dead_inference_consumer(reference) -> None:
    """An unaged gauge would keep exporting the last window forever, reading as live traffic."""
    now = [100.0]
    metrics = MonitorMetrics(CollectorRegistry(), clock=lambda: now[0])
    settings = _settings(sample_every=1, window_frames=1)
    box = Box(cls="person", conf=0.5, xywhn=(0.1, 0.1, 0.1, 0.1))

    _drive(settings, reference, metrics, [_detection(boxes=[box], index=0)])
    assert metrics.registry.get_sample_value("prediction_window_age_seconds") == 0.0

    now[0] = 160.0
    assert metrics.registry.get_sample_value("prediction_window_age_seconds") == 60.0


# --- what Prometheus scrapes ---


def test_every_score_is_exported_beside_its_own_threshold(reference, metrics) -> None:
    """The on-call question is "how close to the bar", not "did it cross"."""
    settings = _settings(sample_every=1, window_frames=1)
    _drive(settings, reference, metrics, [_frame(index=0)])

    for statistic in ("brightness", "contrast", "blur"):
        labels = {"statistic": statistic}
        assert metrics.registry.get_sample_value("drift_score", labels) is not None
        assert metrics.registry.get_sample_value("drift_threshold", labels) > 0


def test_windows_are_counted_by_verdict(reference, metrics) -> None:
    settings = _settings(sample_every=1, window_frames=1)
    frames = [_frame(index=0), _frame(DARK, index=1)]

    _drive(settings, reference, metrics, frames)

    assert metrics.registry.get_sample_value("drift_windows_total", {"verdict": "ok"}) == 1.0
    assert metrics.registry.get_sample_value("drift_windows_total", {"verdict": "drifted"}) == 1.0


def test_episodes_are_counted(reference, metrics) -> None:
    settings = _settings(sample_every=1, window_frames=1)

    _drive(settings, reference, metrics, [_frame(DARK, index=i) for i in range(3)])

    assert metrics.registry.get_sample_value("drift_episodes_total") == 1.0


def test_skipped_frames_are_counted(reference, metrics) -> None:
    settings = _settings(sample_every=1, window_frames=1)

    _drive(settings, reference, metrics, [_frame(b"not an image", index=0)])

    assert metrics.registry.get_sample_value("drift_frames_skipped_total") == 1.0


def test_the_window_age_reports_a_stalled_monitor(reference) -> None:
    """A monitor whose stream died must read as stalled, not as calm water."""
    now = [100.0]
    metrics = MonitorMetrics(CollectorRegistry(), clock=lambda: now[0])
    settings = _settings(sample_every=1, window_frames=1)

    now[0] = 130.0
    assert metrics.registry.get_sample_value("drift_window_age_seconds") == 30.0  # since startup

    _drive(settings, reference, metrics, [_frame(index=0)])
    assert metrics.registry.get_sample_value("drift_window_age_seconds") == 0.0

    now[0] = 175.0
    assert metrics.registry.get_sample_value("drift_window_age_seconds") == 45.0


# --- bounded runs ---


def test_max_messages_bounds_the_loop(reference, metrics) -> None:
    settings = _settings(sample_every=1, window_frames=1)
    frames = [_frame(index=i) for i in range(10)]

    counts = run_loop(settings, reference, _source(frames), metrics, max_messages=3)

    assert counts["consumed"] == 3


# --- startup ---


def test_a_baseline_that_yields_no_thresholds_is_a_named_refusal(reference) -> None:
    """Resolution rejects only an *empty* baseline; one scene has no spread to derive a bar from."""
    single = [BaselineScene(sequence="only", n_frames=9, means=profile_drift_bytes(NORMAL))]

    with pytest.raises(StartupError) as exc:
        monitor.drift_reference(single, Settings())

    message = str(exc.value)
    assert "make profile" in message  # the fixing command, not the module's ValueError
    assert "spread" in message


def test_the_margin_setting_widens_the_derived_bars(reference) -> None:
    means = profile_drift_bytes(NORMAL)
    scenes = [
        BaselineScene(sequence=f"seq{i}", n_frames=10, means={k: v * f for k, v in means.items()})
        for i, f in enumerate((0.9, 1.0, 1.1))
    ]

    wide = monitor.drift_reference(scenes, _settings(threshold_margin=2.0))

    assert wide.thresholds["brightness"] == pytest.approx(2 * reference.thresholds["brightness"])


def test_main_exits_2_when_the_champion_carries_no_baseline(monkeypatch, caplog) -> None:
    """A misconfigured registry is an error in seconds with the fix, not silence for hours."""

    def refuse(settings):
        raise StartupError("the champion's training run carries no 'data.version' tag")

    monkeypatch.setattr(monitor, "resolve_baseline", refuse)

    with caplog.at_level("ERROR"):
        assert monitor.main([]) == 2
    assert "data.version" in caplog.text
