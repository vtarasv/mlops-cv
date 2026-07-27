"""Unit tests for the episode-scoped windowed-count rule."""

from __future__ import annotations

from mlops_cv.streaming.anomaly import EpisodeRule
from mlops_cv.streaming.messages import Box, DetectionEvent, ModelInfo


def _event(sequence: str, frame_index: int, ts_ms: int, n_person: int) -> DetectionEvent:
    boxes = [Box(cls="person", conf=0.9, xywhn=(0.5, 0.5, 0.1, 0.1))] * n_person
    boxes += [Box(cls="vehicle", conf=0.9, xywhn=(0.5, 0.5, 0.2, 0.2))]  # never counted
    return DetectionEvent(
        sequence=sequence,
        frame_index=frame_index,
        ts_frame_ms=ts_ms,
        ts_infer_ms=ts_ms + 20,
        latency_ms=20.0,
        model=ModelInfo(name="stub", version="0"),
        boxes=boxes,
    )


def _rule(threshold: float = 2.5, window_s: float = 5.0) -> EpisodeRule:
    return EpisodeRule(cls="person", window_s=window_s, threshold=threshold)


def test_fires_once_when_crossing_and_stays_silent_while_high() -> None:
    rule = _rule()
    assert rule.observe(_event("seq", 1, 1_000, 1)) is None  # mean 1.0
    alert = rule.observe(_event("seq", 2, 2_000, 5))  # mean 3.0 -> crossing
    assert alert is not None
    assert alert.observed > 2.5
    assert (alert.sequence, alert.frame_index) == ("seq", 2)
    # The condition persists: still one episode, no alert spam.
    assert rule.observe(_event("seq", 3, 3_000, 5)) is None
    assert rule.observe(_event("seq", 4, 4_000, 5)) is None


def test_rearms_after_clearing_and_fires_again() -> None:
    rule = _rule(window_s=2.0)
    assert rule.observe(_event("seq", 1, 1_000, 5)) is not None  # first episode
    # Low counts push the mean back under the threshold -> episode clears silently.
    assert rule.observe(_event("seq", 2, 2_000, 1)) is None
    assert rule.observe(_event("seq", 3, 4_000, 1)) is None  # old highs evicted too
    # Next crossing is a NEW episode.
    assert rule.observe(_event("seq", 4, 5_000, 5)) is not None


def test_duplicate_events_do_not_double_count() -> None:
    rule = _rule(threshold=3.5)
    assert rule.observe(_event("seq", 1, 1_000, 2)) is None  # mean 2.0
    # At-least-once replay: the same (sequence, frame_index) arrives again. Counting it
    # twice would push the mean to (2+2+4)/3 > 3.5 spuriously once frame 2 lands.
    assert rule.observe(_event("seq", 1, 1_000, 2)) is None
    assert rule.observe(_event("seq", 2, 2_000, 4)) is None  # true mean 3.0, no fire
    assert rule.observe(_event("seq", 3, 3_000, 6)) is not None  # genuine crossing


def test_windows_over_frame_time_not_arrival_order() -> None:
    rule = _rule(threshold=4.5, window_s=2.0)
    assert rule.observe(_event("seq", 1, 1_000, 5)) is not None  # episode opens
    assert rule.observe(_event("seq", 2, 1_500, 5)) is None
    # A frame far in scene-time future evicts both highs; its own count is low.
    assert rule.observe(_event("seq", 9, 10_000, 1)) is None  # mean 1.0 -> cleared
    assert rule.observe(_event("seq", 10, 10_500, 5)) is None  # mean 3.0, under threshold
    assert rule.observe(_event("seq", 11, 10_600, 9)) is not None  # new episode


def test_alert_payload_records_rule_and_reading() -> None:
    alert = _rule(threshold=0.5, window_s=5.0).observe(_event("seq", 7, 1_000, 2))
    assert alert is not None
    assert alert.rule == "windowed-count"
    assert alert.cls == "person"
    assert alert.threshold == 0.5
    assert alert.window_s == 5.0
    assert alert.observed == 2.0
    assert alert.n_frames == 1
    assert alert.ts_ms == 1_000
    assert alert.kafka_key() == b"seq"
