"""The inference consumer's production wiring: the serving detector behind the seam."""

from __future__ import annotations

import _thread
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from mlops_cv.config import Settings
from mlops_cv.config.settings import ServingSettings
from mlops_cv.serving.runtime import Detector
from mlops_cv.startup import StartupError
from mlops_cv.streaming import inference_consumer
from mlops_cv.streaming.messages import Box, DetectionEvent, ModelInfo, pack_frame

MODEL = ModelInfo(name="aerial-object-detector", version="8")
BOX = Box(cls="vehicle", conf=0.9, xywhn=(0.5, 0.5, 0.1, 0.1))


class FakeFrame:
    """A consumed Frame message on partition 0; its JPEG is the frame index as bytes."""

    def __init__(self, offset: int) -> None:
        self._offset = offset
        self._key, self._headers = pack_frame("seq", frame_index=offset, ts_ms=0)

    def key(self) -> bytes:
        return self._key

    def headers(self):
        return self._headers

    def value(self) -> bytes:
        return str(self._offset).encode()

    def topic(self) -> str:
        return "frames"

    def partition(self) -> int:
        return 0

    def offset(self) -> int:
        return self._offset

    def error(self) -> None:
        return None


@pytest.fixture
def kafka(monkeypatch):
    """In-memory Consumer/Producer: frames to poll, events produced, offsets stored."""
    bus = SimpleNamespace(frames=[], events=[], stored=[])

    class Consumer:
        def __init__(self, config) -> None:
            pass

        def subscribe(self, topics) -> None:
            pass

        def poll(self, timeout):
            return bus.frames.pop(0) if bus.frames else None

        def store_offsets(self, offsets) -> None:
            bus.stored.extend(offsets)

        def assignment(self):
            return []

        def close(self) -> None:
            pass

    class Producer:
        def __init__(self, config) -> None:
            self.pending: list = []

        def __len__(self) -> int:
            return len(self.pending)

        def produce(self, topic, value, key, on_delivery) -> None:
            bus.events.append(DetectionEvent.from_value(value).frame_index)
            self.pending.append(on_delivery)

        def poll(self, timeout) -> None:
            while self.pending:
                self.pending.pop(0)(None, None)

        flush = poll

    module = SimpleNamespace(
        Consumer=Consumer, Producer=Producer, TopicPartition=lambda t, p, offset: offset
    )
    monkeypatch.setitem(sys.modules, "confluent_kafka", module)
    return bus


def _run(bus, infer, frames: int, max_inflight: int) -> dict[str, int]:
    bus.frames = [FakeFrame(offset) for offset in range(frames)]
    settings = Settings(serving=ServingSettings(max_inflight=max_inflight))
    return inference_consumer.run_loop(settings, infer, MODEL, max_messages=frames)


def test_detect_answers_with_wire_boxes(fake_session, png_upload) -> None:
    """The production seam: bytes in, the detector's graph-named wire boxes out."""
    # One detection in letterbox pixels; a 640x640 source maps 1:1.
    output = np.array([[[100.0, 100.0, 200.0, 200.0, 0.9, 1.0]]], dtype=np.float32)
    detector = Detector(fake_session(output), MODEL)

    infer = partial(detector.detect, conf_threshold=0.25)
    boxes = infer(png_upload(640, 640))

    assert len(boxes) == 1
    assert boxes[0].cls == "vehicle"  # the graph's own class map, not a configured one
    assert boxes[0].conf == pytest.approx(0.9)


def test_detect_binds_the_confidence_floor(fake_session, png_upload) -> None:
    """The seam takes only bytes, so the floor is bound at wiring time — from settings."""
    output = np.zeros((1, 2, 6), dtype=np.float32)
    output[0, :, :4] = [100, 100, 200, 200]
    output[0, :, 4] = [0.9, 0.3]
    detector = Detector(fake_session(output), MODEL)

    assert len(detector.detect(png_upload(640, 640), 0.25)) == 2
    assert len(detector.detect(png_upload(640, 640), 0.5)) == 1


def test_detect_raises_on_undecodable_bytes(fake_session) -> None:
    """Poison stays poison: the loop's skip-with-stored-offset path needs the raise."""
    empty = np.zeros((1, 0, 6), dtype=np.float32)
    infer = partial(Detector(fake_session(empty), MODEL).detect, conf_threshold=0.25)
    with pytest.raises(ValueError, match="decode"):
        infer(b"not a jpeg")


def test_main_exits_2_when_the_registry_cannot_serve(monkeypatch, caplog) -> None:
    """No champion / no published graph is an ordinary state: an exit code and the fixing
    command, exactly like the service — never a traceback into the container log."""
    import mlops_cv.serving.resolve as serving_resolve

    def refuse(settings):
        raise StartupError("no champion — run `make train`")

    monkeypatch.setattr(serving_resolve, "resolve_graph", refuse)

    with caplog.at_level("ERROR"):
        assert inference_consumer.main([]) == 2
    assert "no champion" in caplog.text


def test_main_serves_the_resolved_detector_at_the_settings_floor(monkeypatch) -> None:
    """main wires the loop with the detector's own identity and the settings threshold."""
    import mlops_cv.serving.resolve as serving_resolve
    from mlops_cv.config import get_settings

    calls: dict = {}

    class StubDetector:
        model = MODEL

        def detect(self, jpeg: bytes, conf_threshold: float):
            calls["conf"] = conf_threshold
            return []

    monkeypatch.setattr(
        serving_resolve,
        "resolve_graph",
        lambda settings: (None, MODEL),  # type: ignore
    )
    monkeypatch.setattr(Detector, "load", classmethod(lambda cls, path, model: StubDetector()))

    def fake_loop(settings, infer, model, **kwargs):
        calls["model"] = model
        infer(b"jpeg-bytes")  # drive the seam so the bound threshold is observable
        return {}

    monkeypatch.setattr(inference_consumer, "run_loop", fake_loop)

    assert inference_consumer.main([]) == 0
    assert calls["model"] == MODEL
    assert calls["conf"] == get_settings().serving.conf_threshold


def test_events_and_offsets_follow_poll_order_when_a_later_frame_finishes_first(kafka) -> None:
    """Frame 0 finishes only after frame 1 has: its event is still produced first."""
    second_done = threading.Event()

    def infer(jpeg: bytes) -> list[Box]:
        if jpeg == b"0":
            assert second_done.wait(timeout=2), "frame 1 never ran alongside frame 0"
        else:
            second_done.set()
        return [BOX]

    counts = _run(kafka, infer, frames=2, max_inflight=2)

    assert counts["events"] == 2
    assert kafka.events == [0, 1]
    assert kafka.stored == [1, 2]


def test_a_poison_frame_inside_the_window_is_skipped_in_order(kafka) -> None:
    """Frame 1 fails while frame 0 is still running: skipped, offsets still advance in order."""
    poison_failed = threading.Event()

    def infer(jpeg: bytes) -> list[Box]:
        if jpeg == b"1":
            poison_failed.set()
            raise ValueError("cannot decode image bytes")
        if jpeg == b"0":
            assert poison_failed.wait(timeout=2), "the poison frame never ran alongside frame 0"
        return [BOX]

    counts = _run(kafka, infer, frames=3, max_inflight=2)

    assert counts == {"consumed": 3, "events": 2, "skipped": 1, "delivery_failures": 0}
    assert kafka.events == [0, 2]
    assert kafka.stored == sorted(kafka.stored)
    assert kafka.stored[-1] == 3


def test_at_most_max_inflight_detects_run_at_once(kafka) -> None:
    """A third frame waits for a slot while two detects block."""
    gate, entered = threading.Event(), queue.Queue()

    def infer(jpeg: bytes) -> list[Box]:
        entered.put(jpeg)
        gate.wait(timeout=10)
        return [BOX]

    with ThreadPoolExecutor(1) as runner:
        try:
            loop = runner.submit(_run, kafka, infer, 3, 2)
            for _ in range(2):
                entered.get(timeout=5)
            time.sleep(0.3)
            assert entered.empty(), "a third detect ran past max_inflight"
        finally:
            gate.set()
        assert loop.result(timeout=10)["events"] == 3
    assert kafka.events == [0, 1, 2]


def test_one_in_flight_runs_one_frame_at_a_time(kafka) -> None:
    """N = 1 is the sequential loop: one detect at a time, poison skipped in order."""
    running, peak = [0], [0]

    def infer(jpeg: bytes) -> list[Box]:
        running[0] += 1
        peak[0] = max(peak[0], running[0])
        time.sleep(0.01)
        running[0] -= 1
        if jpeg == b"1":
            raise ValueError("cannot decode image bytes")
        return [BOX]

    counts = _run(kafka, infer, frames=3, max_inflight=1)

    assert peak[0] == 1
    assert counts == {"consumed": 3, "events": 2, "skipped": 1, "delivery_failures": 0}
    assert kafka.events == [0, 2]
    assert kafka.stored == [1, 2, 3]


def test_ctrl_c_while_waiting_on_the_oldest_still_emits_it(kafka) -> None:
    """The interrupt lands in the wait on frame 0: the drain still emits 0 then 1."""
    second_entered = threading.Event()

    def infer(jpeg: bytes) -> list[Box]:
        if jpeg == b"0":
            assert second_entered.wait(timeout=2)
            time.sleep(0.1)  # let the loop block on this frame's result
            _thread.interrupt_main()
            time.sleep(0.1)
        else:
            second_entered.set()
        return [BOX]

    counts = _run(kafka, infer, frames=3, max_inflight=2)

    assert counts["events"] == 2
    assert kafka.events == [0, 1]
    assert kafka.stored == [1, 2]
