"""Stream view: the pairing, the event JSON and the web handler; no Kafka, no network."""

from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer

import pytest

from mlops_cv.streaming import view
from mlops_cv.streaming.messages import Box, DetectionEvent, FrameRef, ModelInfo

SEQ = "uav1-20261018T120000Z"


def ref(index: int, sequence: str = SEQ, ts_ms: int = 0) -> FrameRef:
    return FrameRef(sequence=sequence, frame_index=index, ts_ms=ts_ms or 1_000 + 33 * index)


def event(index: int, sequence: str = SEQ, conf: float = 0.8) -> DetectionEvent:
    return DetectionEvent(
        sequence=sequence,
        frame_index=index,
        ts_frame_ms=1_000 + 33 * index,
        ts_infer_ms=1_020 + 33 * index,
        latency_ms=12.5,
        model=ModelInfo(name="aerial-object-detector", version="7"),
        boxes=[Box(cls="person", conf=conf, xywhn=(0.5, 0.5, 0.1, 0.2))],
    )


def test_a_detection_pairs_with_its_frame_by_address() -> None:
    pairing = view.Pairing()
    pairing.frame(ref(1), b"jpeg-1")
    pairing.frame(ref(2), b"jpeg-2")
    pair = pairing.detection(event(1))
    assert pair is not None
    assert (pair[0].frame_index, pair[1]) == (1, b"jpeg-1")


def test_a_detection_before_its_frame_is_skipped() -> None:
    pairing = view.Pairing()
    assert pairing.detection(event(1)) is None
    pairing.frame(ref(1), b"jpeg-1")
    assert pairing.detection(event(2)) is None


def test_pending_frames_stay_bounded_at_the_cap() -> None:
    pairing = view.Pairing()
    for index in range(view.PENDING_CAP + 10):
        pairing.frame(ref(index), b"jpeg")
    assert len(pairing.pending) == view.PENDING_CAP
    assert pairing.detection(event(9)) is None  # evicted, oldest first
    assert pairing.detection(event(10)) is not None


def test_a_later_duplicate_frame_or_event_wins() -> None:
    pairing = view.Pairing()
    pairing.frame(ref(1), b"first")
    pairing.frame(ref(1), b"replayed")
    first = pairing.detection(event(1, conf=0.4))
    later = pairing.detection(event(1, conf=0.9))
    assert first is not None and later is not None
    assert first[1] == b"replayed"
    assert later[0].boxes[0].conf == 0.9


def test_a_new_sequence_takes_over_and_counts_a_session() -> None:
    pairing = view.Pairing()
    pairing.frame(ref(1, "old"), b"old")
    pairing.frame(ref(1, "new"), b"new")
    assert (pairing.sequence, pairing.sessions) == ("new", 2)
    assert pairing.detection(event(1, "old")) is None
    assert pairing.detection(event(1, "new")) is not None


def test_an_older_live_sequence_does_not_take_back_over() -> None:
    pairing = view.Pairing()
    for index in range(1, 4):  # two links at once: the newest stays followed
        pairing.frame(ref(index, "old"), b"old")
        pairing.frame(ref(index, "new"), b"new")
    assert (pairing.sequence, pairing.sessions) == ("new", 2)


def test_a_seen_sequence_takes_over_once_the_followed_one_stalls() -> None:
    pairing = view.Pairing()
    pairing.frame(ref(1, "a", ts_ms=1_000), b"a")
    pairing.frame(ref(1, "b", ts_ms=2_000), b"b")
    pairing.frame(ref(1, "a", ts_ms=2_000 + view.STALE_MS + 1), b"a again")  # a demo loop wraps
    assert (pairing.sequence, pairing.sessions) == ("a", 3)


def test_a_pinned_sequence_ignores_every_other() -> None:
    pairing = view.Pairing(sequence="a")
    pairing.frame(ref(1, "a"), b"a")
    pairing.frame(ref(1, "b", ts_ms=999_999), b"b")
    assert (pairing.sequence, pairing.sessions) == ("a", 1)
    assert pairing.detection(event(1, "b")) is None


def test_index_gaps_count_the_dropped_frames() -> None:
    pairing = view.Pairing()
    for index in (1, 2, 5, 6, 10, 1):  # the last one is a replay wrapping, not a gap
        pairing.frame(ref(index), b"jpeg")
    assert pairing.gaps == 2 + 3


def test_the_event_json_carries_the_frame_and_its_boxes() -> None:
    jpeg = bytes(range(256))  # not valid UTF-8: only base64 survives JSON
    data = json.loads(view.event_json(event(3), jpeg, gaps=2, sessions=1))
    assert set(data) == {
        "sequence",
        "frame_index",
        "ts_frame_ms",
        "latency_ms",
        "model",
        "boxes",
        "jpeg_b64",
        "gaps",
        "sessions",
    }
    assert base64.b64decode(data["jpeg_b64"]) == jpeg
    assert (data["sequence"], data["frame_index"], data["ts_frame_ms"]) == (SEQ, 3, 1_099)
    assert data["model"] == {"name": "aerial-object-detector", "version": "7"}
    assert data["boxes"] == [{"cls": "person", "conf": 0.8, "xywhn": [0.5, 0.5, 0.1, 0.2]}]
    assert (data["latency_ms"], data["gaps"], data["sessions"]) == (12.5, 2, 1)


@pytest.fixture
def server(monkeypatch) -> Iterator[str]:
    monkeypatch.setattr(view.Handler, "page", view.page(0.25, ["person", "vehicle"]))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), view.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_the_page_is_html_with_the_floor_and_classes(server: str) -> None:
    with urllib.request.urlopen(f"{server}/") as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/html")
        body = resp.read()
    assert b"new EventSource('/events')" in body
    assert b'const FLOOR=0.25,CLASSES=["person", "vehicle"]' in body


def test_an_unknown_path_is_404(server: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(f"{server}/nope")
    assert err.value.code == 404


def test_a_new_client_gets_the_latest_event_first(server: str, monkeypatch) -> None:
    monkeypatch.setattr(view, "event_count", 0)
    view.publish('{"frame_index": 1}')
    view.publish('{"frame_index": 2}')
    with urllib.request.urlopen(f"{server}/events", timeout=5) as resp:
        assert resp.headers["Content-Type"] == "text/event-stream"
        assert resp.readline() == b'data: {"frame_index": 2}\n'


def test_gaps_count_per_followed_sequence() -> None:
    pairing = view.Pairing()
    pairing.frame(ref(1, "old"), b"old")
    pairing.frame(ref(5, "old"), b"old")
    pairing.frame(ref(1, "new"), b"new")
    assert pairing.gaps == 0
