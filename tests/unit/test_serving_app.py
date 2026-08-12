"""The service at its highest seam: HTTP in, JSON out, with only the runtime faked.

The app is built on a real :class:`Detector` over a fake session, so a request exercises
decode, letterbox, rescale and the wire-contract conversion — the response shape asserted
here is the one a client actually receives.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlops_cv.config import Settings
from mlops_cv.serving.app import create_app
from mlops_cv.serving.runtime import Detector
from mlops_cv.serving.schemas import Detections, Health
from mlops_cv.streaming.messages import Box, ModelInfo

MODEL = ModelInfo(name="aerial-object-detector", version="7")

# Three detections at descending confidence, in letterbox pixels of a 640x640 upload.
OUTPUT = np.zeros((1, 3, 6), dtype=np.float32)
OUTPUT[0, :, :4] = [160.0, 160.0, 320.0, 320.0]
OUTPUT[0, :, 4] = [0.9, 0.4, 0.1]
OUTPUT[0, :, 5] = [0, 1, 2]


@pytest.fixture
def client(fake_session) -> TestClient:
    return TestClient(create_app(Detector(fake_session(OUTPUT), MODEL), Settings()))


@pytest.fixture
def upload(png_upload) -> dict:
    return {"image": ("frame.png", png_upload(640, 640), "image/png")}


def test_health_reports_the_loaded_model(client: TestClient) -> None:
    """What is running is answerable without reading logs — the rollout check."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == {"name": "aerial-object-detector", "version": "7"}
    assert body["imgsz"] == 640


def test_predict_answers_with_detections_and_their_model(client: TestClient, upload) -> None:
    response = client.post("/predict", files=upload)
    assert response.status_code == 200
    body = response.json()

    assert body["model"] == {"name": "aerial-object-detector", "version": "7"}
    assert body["latency_ms"] > 0
    # Default threshold 0.25 keeps the two above it.
    assert [box["cls"] for box in body["boxes"]] == ["person", "vehicle"]
    first = body["boxes"][0]
    assert set(first) == {"cls", "conf", "xywhn"}  # the wire contract's Box, verbatim
    assert first["conf"] == pytest.approx(0.9)
    assert first["xywhn"] == pytest.approx([0.375, 0.375, 0.25, 0.25])


def test_predict_honors_a_per_request_confidence_threshold(client: TestClient, upload) -> None:
    """Precision/recall is a client's call — no redeploy to change it."""
    assert len(client.post("/predict", files=upload, params={"conf": 0.05}).json()["boxes"]) == 3
    assert len(client.post("/predict", files=upload, params={"conf": 0.5}).json()["boxes"]) == 1
    assert client.post("/predict", files=upload, params={"conf": 0.99}).json()["boxes"] == []


def test_default_threshold_comes_from_settings(fake_session, upload) -> None:
    settings = Settings()
    settings.serving.conf_threshold = 0.5
    app = create_app(Detector(fake_session(OUTPUT), MODEL), settings)
    boxes = TestClient(app).post("/predict", files=upload).json()["boxes"]
    assert [box["cls"] for box in boxes] == ["person"]


def test_out_of_range_threshold_is_rejected(client: TestClient, upload) -> None:
    assert client.post("/predict", files=upload, params={"conf": 1.5}).status_code == 422


def test_undecodable_upload_is_a_client_error(client: TestClient) -> None:
    files = {"image": ("frame.png", b"definitely not an image", "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "decode" in response.json()["detail"]


def test_missing_upload_is_a_client_error(client: TestClient) -> None:
    assert client.post("/predict").status_code == 422


def test_a_wrong_graph_is_not_blamed_on_the_client(fake_session, upload) -> None:
    """Only the decode is a 400.

    The pre/post module raises ValueError for a malformed graph *output* too; answering
    that with a 400 would file a broken deployment under client errors in the very
    metrics that are supposed to reveal it.
    """
    wrong_shape = np.zeros((1, 3, 5), dtype=np.float32)  # not (max_det, 6)
    app = create_app(Detector(fake_session(wrong_shape), MODEL), Settings())
    with pytest.raises(ValueError, match="max_det, 6"):
        TestClient(app).post("/predict", files=upload)


def test_metrics_are_exposed_in_prometheus_format(client: TestClient, upload) -> None:
    """Request metrics are scrapeable without log-tailing.

    Asserted on the exposition, not on a per-handler label: the instrumentator collects
    into the process-global prometheus registry, so in a test session that builds several
    apps only the first one's collectors stay registered. One app per process in
    production — but a label assertion here would pass or fail by test ordering.
    """
    client.post("/predict", files=upload)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "http_requests_total" in response.text


def test_the_service_imports_without_the_heavy_runtimes() -> None:
    """The deployment artifact carries no torch/ultralytics, and loads no CUDA to import.

    Both the registry client and onnxruntime are imported where they are used, so building
    the app costs neither — the same property CI relies on to run these tests.
    """
    check = (
        "import sys; import mlops_cv.serving.serve; "
        "banned = {'torch', 'ultralytics', 'onnxruntime', 'mlflow', 'PIL'} & set(sys.modules); "
        "assert not banned, banned"
    )
    subprocess.run([sys.executable, "-c", check], check=True)


def test_response_models_reuse_the_wire_contract(fake_session) -> None:
    """Pinned like the contract tests: HTTP and Kafka consumers share one vocabulary.

    A copied Box/ModelInfo would drift silently — the two would still serialize the same
    on the day it was copied.
    """
    assert Detections.model_fields["boxes"].annotation == list[Box]
    assert Detections.model_fields["model"].annotation is ModelInfo
    assert Health.model_fields["model"].annotation is ModelInfo
