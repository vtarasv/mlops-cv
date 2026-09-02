"""The inference consumer's production wiring: the serving detector behind the seam."""

from __future__ import annotations

import numpy as np
import pytest

from mlops_cv.serving.runtime import Detector
from mlops_cv.startup import StartupError
from mlops_cv.streaming import inference_consumer
from mlops_cv.streaming.messages import ModelInfo

MODEL = ModelInfo(name="aerial-object-detector", version="8")


def test_detector_inference_answers_with_wire_boxes(fake_session, png_upload) -> None:
    """The production seam: bytes in, the detector's graph-named wire boxes out."""
    # One detection in letterbox pixels; a 640x640 source maps 1:1.
    output = np.array([[[100.0, 100.0, 200.0, 200.0, 0.9, 1.0]]], dtype=np.float32)
    detector = Detector(fake_session(output), MODEL)

    infer = inference_consumer.detector_inference(detector, conf_threshold=0.25)
    boxes = infer(png_upload(640, 640))

    assert len(boxes) == 1
    assert boxes[0].cls == "vehicle"  # the graph's own class map, not a configured one
    assert boxes[0].conf == pytest.approx(0.9)


def test_detector_inference_binds_the_confidence_floor(fake_session, png_upload) -> None:
    """The seam takes only bytes, so the floor is bound at wiring time — from settings."""
    output = np.zeros((1, 2, 6), dtype=np.float32)
    output[0, :, :4] = [100, 100, 200, 200]
    output[0, :, 4] = [0.9, 0.3]
    detector = Detector(fake_session(output), MODEL)

    assert len(inference_consumer.detector_inference(detector, 0.25)(png_upload(640, 640))) == 2
    assert len(inference_consumer.detector_inference(detector, 0.5)(png_upload(640, 640))) == 1


def test_detector_inference_raises_on_undecodable_bytes(fake_session) -> None:
    """Poison stays poison: the loop's skip-with-stored-offset path needs the raise."""
    empty = np.zeros((1, 0, 6), dtype=np.float32)
    infer = inference_consumer.detector_inference(Detector(fake_session(empty), MODEL), 0.25)
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
    from mlops_cv.serving.resolve import ResolvedGraph

    calls: dict = {}

    class StubDetector:
        model = MODEL

        def detect(self, jpeg: bytes, conf_threshold: float):
            calls["conf"] = conf_threshold
            return []

    monkeypatch.setattr(
        serving_resolve,
        "resolve_graph",
        lambda settings: ResolvedGraph(path=None, model=MODEL),  # type: ignore
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
