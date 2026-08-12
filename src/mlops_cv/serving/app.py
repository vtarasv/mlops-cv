"""The HTTP detection service."""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator

from mlops_cv.config import Settings
from mlops_cv.serving.prepost import DecodeError
from mlops_cv.serving.runtime import Detector
from mlops_cv.serving.schemas import Detections, Health

logger = logging.getLogger(__name__)


def create_app(detector: Detector, settings: Settings) -> FastAPI:
    """The service over one loaded detector."""
    default_conf = settings.serving.conf_threshold

    app = FastAPI(
        title="Aerial object detection",
        description="Detections from the champion model's published graph.",
        version=detector.model.version,
    )
    Instrumentator().instrument(app).expose(app)

    @app.get("/health")
    def health() -> Health:
        return Health(status="ok", model=detector.model, imgsz=detector.imgsz)

    @app.post("/predict")
    async def predict(
        image: Annotated[UploadFile, File(description="the frame to detect in")],
        conf: Annotated[
            float, Query(ge=0.0, le=1.0, description="minimum detection confidence")
        ] = default_conf,
    ) -> Detections:
        """Detect in one uploaded image."""
        payload = await image.read()
        started = time.perf_counter()
        try:
            boxes = detector.detect(payload, conf_threshold=conf)
        except DecodeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Detections(
            model=detector.model,
            boxes=boxes,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    return app
