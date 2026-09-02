"""The detector seam: an injected runtime session, real pre/post math around it.

The session is the only fake — everything between the uploaded bytes and the returned
boxes is the code the service runs, so the wiring (input tensor, box rescale, class-name
merge, normalization) is exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlops_cv.serving.runtime import Detector, graph_spec
from mlops_cv.startup import StartupError
from mlops_cv.streaming.messages import ModelInfo

MODEL = ModelInfo(name="aerial-object-detector", version="7")
EMPTY = np.zeros((1, 0, 6), dtype=np.float32)


def test_graph_spec_reads_class_names_and_resolution_from_the_graph(
    fake_session, graph_names
) -> None:
    """Names travel inside the published graph, so serving cannot drift from the model."""
    spec = graph_spec(fake_session(EMPTY))
    assert spec.names == graph_names
    assert spec.imgsz == 640


def test_graph_without_class_names_refuses_to_serve(fake_session) -> None:
    with pytest.raises(StartupError, match="class names"):
        graph_spec(fake_session(EMPTY, {"imgsz": "[640, 640]"}))


def test_unreadable_graph_metadata_refuses_to_serve(fake_session, graph_metadata) -> None:
    """A startup failure, not a traceback out of ast.literal_eval mid-request."""
    with pytest.raises(StartupError, match="unreadable"):
        graph_spec(fake_session(EMPTY, graph_metadata | {"names": "{0: person"}))


def test_non_square_graph_input_refuses_to_serve(fake_session, graph_metadata) -> None:
    """The letterbox targets a square; a rectangular graph would be silently mis-served."""
    with pytest.raises(StartupError, match="square"):
        graph_spec(fake_session(EMPTY, graph_metadata | {"imgsz": "[640, 480]"}))


def test_detect_feeds_the_graphs_input_tensor(fake_session, png_upload) -> None:
    session = fake_session(EMPTY)
    Detector(session, MODEL).detect(png_upload(720, 1280), conf_threshold=0.25)

    (feed,) = session.feeds
    tensor = feed["images"]
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0


def test_detect_returns_wire_boxes_in_normalized_source_coordinates(
    fake_session, png_upload
) -> None:
    # 1280x720 source -> gain 0.5, pads (0, 140): letterbox (320, 250, 480, 400)
    # rescales to source (640, 220, 960, 520).
    output = np.array([[[320.0, 250.0, 480.0, 400.0, 0.9, 1.0]]], dtype=np.float32)
    boxes = Detector(fake_session(output), MODEL).detect(png_upload(720, 1280), conf_threshold=0.25)

    assert len(boxes) == 1
    box = boxes[0]
    assert box.cls == "vehicle"  # merged name, not the raw class id
    assert box.conf == pytest.approx(0.9)
    assert box.xywhn == pytest.approx((800 / 1280, 370 / 720, 320 / 1280, 300 / 720))
    assert all(0.0 <= v <= 1.0 for v in box.xywhn)


def test_detect_applies_the_requested_confidence_threshold(fake_session, png_upload) -> None:
    output = np.zeros((1, 3, 6), dtype=np.float32)
    output[0, :, :4] = [100, 100, 200, 200]
    output[0, :, 4] = [0.9, 0.4, 0.1]
    output[0, :, 5] = [0, 1, 2]
    detector = Detector(fake_session(output), MODEL)
    upload = png_upload(640, 640)

    assert len(detector.detect(upload, conf_threshold=0.25)) == 2
    assert len(detector.detect(upload, conf_threshold=0.5)) == 1
    assert detector.detect(upload, conf_threshold=0.95) == []


def test_detect_rejects_an_undecodable_upload(fake_session) -> None:
    with pytest.raises(ValueError, match="decode"):
        Detector(fake_session(EMPTY), MODEL).detect(b"not an image", conf_threshold=0.25)


def test_detector_reports_the_model_it_loaded(fake_session) -> None:
    detector = Detector(fake_session(EMPTY), MODEL)
    assert detector.model == MODEL
    assert detector.imgsz == 640
