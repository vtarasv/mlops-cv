"""What the service answers with."""

from __future__ import annotations

from pydantic import BaseModel

from mlops_cv.streaming.messages import Box, ModelInfo


class Detections(BaseModel):
    """One answer: what was found, by which model, and how long the answer took."""

    model: ModelInfo
    boxes: list[Box]
    latency_ms: float  # inference wall time, decode through box conversion


class Health(BaseModel):
    """What is running: the identity of the loaded model and the size it takes."""

    status: str
    model: ModelInfo
    imgsz: int
