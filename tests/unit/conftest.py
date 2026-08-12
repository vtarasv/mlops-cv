"""Shared doubles for the serving tests: an onnxruntime-shaped session and PNG uploads.

The runtime tests drive the detector directly, the app tests drive it through HTTP; both
need the same fake session.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

# The merged VisDrone class map, as the exporter embeds it in the graph metadata.
GRAPH_NAMES = {0: "person", 1: "vehicle", 2: "two-three-wheeler"}
GRAPH_METADATA = {
    "names": str(GRAPH_NAMES),
    "imgsz": "[640, 640]",
    "task": "detect",
    "batch": "1",
}


class FakeSession:
    """An onnxruntime session as the detector uses it: metadata, one input, one output."""

    def __init__(self, output: np.ndarray, metadata: dict[str, str] | None = None) -> None:
        self.output = output
        self.metadata = GRAPH_METADATA if metadata is None else metadata
        self.feeds: list[dict[str, np.ndarray]] = []

    def get_modelmeta(self) -> SimpleNamespace:
        return SimpleNamespace(custom_metadata_map=dict(self.metadata))

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="images", shape=[1, 3, 640, 640])]

    def get_providers(self) -> list[str]:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def run(self, output_names, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(feed)
        return [self.output]


@pytest.fixture
def fake_session() -> type[FakeSession]:
    """The session double's class — tests construct it with the output they want back."""
    return FakeSession


@pytest.fixture
def graph_names() -> dict[int, str]:
    """The class map the fake graph advertises."""
    return dict(GRAPH_NAMES)


@pytest.fixture
def graph_metadata() -> dict[str, str]:
    """The fake graph's embedded metadata, for tests that corrupt one field of it."""
    return dict(GRAPH_METADATA)


@pytest.fixture
def png_upload() -> Callable[[int, int], bytes]:
    """``png_upload(height, width)`` -> a lossless gradient PNG, decoding to exactly itself."""

    def encode(height: int, width: int) -> bytes:
        import cv2

        rows = np.linspace(0, 255, height, dtype=np.uint8)[:, None, None]
        cols = np.linspace(0, 255, width, dtype=np.uint8)[None, :, None]
        image = np.broadcast_to(rows // 2 + cols // 2, (height, width, 3)).astype(np.uint8)
        ok, buffer = cv2.imencode(".png", image)
        assert ok
        return buffer.tobytes()

    return encode
