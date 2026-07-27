"""Contract pins for the streaming messages module."""

from __future__ import annotations

import json

import pytest

from mlops_cv.streaming.messages import (
    Alert,
    Box,
    DetectionEvent,
    FrameRef,
    ModelInfo,
    pack_frame,
    parse_frame,
)


def _event(**overrides) -> DetectionEvent:
    base = dict(
        sequence="uav0000161_00000_v",
        frame_index=42,
        ts_frame_ms=1721742000123,
        ts_infer_ms=1721742000145,
        latency_ms=22.1,
        model=ModelInfo(name="aerial-object-detector", version="7"),
        boxes=[Box(cls="person", conf=0.83, xywhn=(0.51, 0.42, 0.03, 0.08))],
    )
    # model_validate, not **kwargs: the mixed-value dict widens to a union the ctor rejects.
    return DetectionEvent.model_validate(base | overrides)


def test_frame_pack_parse_round_trip() -> None:
    key, headers = pack_frame("uav0000161_00000_v", frame_index=42, ts_ms=1721742000123)
    ref = parse_frame(key, headers)
    assert ref == FrameRef(sequence="uav0000161_00000_v", frame_index=42, ts_ms=1721742000123)


def test_frame_key_is_the_sequence() -> None:
    # Contract pin: the message key IS the sequence (per-clip ordering within a partition).
    key, _ = pack_frame("seqA", frame_index=1, ts_ms=1)
    assert key == b"seqA"


def test_frame_headers_are_readable_decimal_strings() -> None:
    # Contract pin: headers stay human-readable in Console (ASCII decimal, stable names).
    _, headers = pack_frame("seqA", frame_index=7, ts_ms=123)
    assert dict(headers) == {"frame_index": b"7", "ts_ms": b"123"}


def test_parse_ignores_unknown_headers() -> None:
    key, headers = pack_frame("seqA", frame_index=7, ts_ms=123)
    ref = parse_frame(key, [*headers, ("x-extra", b"ignored")])
    assert ref.frame_index == 7


def test_parse_rejects_missing_headers() -> None:
    with pytest.raises(ValueError, match="frame_index"):
        parse_frame(b"seqA", [("ts_ms", b"123")])


def test_parse_accepts_the_clients_other_header_shapes() -> None:
    # The Kafka client shares one header type across produce/consume: a mapping, and str
    # values, are both admissible inbound even though pack_frame emits list-of-bytes pairs.
    ref = parse_frame(b"seqA", {"frame_index": "7", "ts_ms": b"123"})
    assert ref == FrameRef(sequence="seqA", frame_index=7, ts_ms=123)


def test_parse_rejects_empty_header_value() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_frame(b"seqA", [("frame_index", None), ("ts_ms", b"123")])


def test_parse_rejects_missing_key() -> None:
    _, headers = pack_frame("seqA", frame_index=7, ts_ms=123)
    with pytest.raises(ValueError, match="key"):
        parse_frame(None, headers)


def test_detection_event_value_round_trip() -> None:
    event = _event()
    assert DetectionEvent.from_value(event.to_value()) == event


def test_detection_event_schema_version_pinned() -> None:
    # Contract pin: readers key their expectations off this number.
    assert _event().schema_version == 1
    assert json.loads(_event().to_value())["schema_version"] == 1


def test_detection_event_wire_field_names_pinned() -> None:
    # Contract pin: these exact field names are what downstream readers (the drift
    # monitor) address — renaming any of them is a breaking protocol change.
    payload = json.loads(_event().to_value())
    assert set(payload) == {
        "schema_version",
        "sequence",
        "frame_index",
        "ts_frame_ms",
        "ts_infer_ms",
        "latency_ms",
        "model",
        "boxes",
    }
    assert set(payload["model"]) == {"name", "version"}
    assert set(payload["boxes"][0]) == {"cls", "conf", "xywhn"}


def test_detection_event_parse_is_skew_tolerant() -> None:
    # A newer producer may add fields; an older reader must not choke on them.
    payload = json.loads(_event().to_value())
    payload["future_field"] = "ignored"
    event = DetectionEvent.from_value(json.dumps(payload).encode())
    assert event.frame_index == 42


def test_detection_event_kafka_key_is_the_sequence() -> None:
    # Contract pin: detection events co-partition with their frames.
    assert _event().kafka_key() == b"uav0000161_00000_v"


def test_detection_event_empty_boxes_is_valid() -> None:
    # "Nothing detected" is an answer — an event with no boxes must round-trip.
    event = _event(boxes=[])
    assert DetectionEvent.from_value(event.to_value()).boxes == []


def _alert() -> Alert:
    return Alert(
        rule="windowed-count",
        cls="person",
        window_s=5.0,
        threshold=40.0,
        observed=52.4,
        n_frames=150,
        sequence="uav0000073_00600_v",
        frame_index=42,
        ts_ms=1721742000123,
    )


def test_alert_value_round_trip_and_key() -> None:
    alert = _alert()
    assert Alert.from_value(alert.to_value()) == alert
    assert alert.kafka_key() == b"uav0000073_00600_v"


def test_alert_wire_field_names_pinned() -> None:
    payload = json.loads(_alert().to_value())
    assert set(payload) == {
        "schema_version",
        "rule",
        "cls",
        "window_s",
        "threshold",
        "observed",
        "n_frames",
        "sequence",
        "frame_index",
        "ts_ms",
    }
    assert payload["schema_version"] == 1


def test_alert_parse_is_skew_tolerant() -> None:
    payload = json.loads(_alert().to_value())
    payload["future_field"] = "ignored"
    assert Alert.from_value(json.dumps(payload).encode()).observed == 52.4
