"""Running the published graph: the onnxruntime session and the boxes it answers with.

The detector is the seam the service is built on — construction takes a *session*, so every
test drives the real pre/post math with only the runtime faked. Around the session it does
exactly what the measured ``onnx-ort`` rung did: letterbox to the graph's resolution,
feed the tensor, rescale the detections back to source pixels, then merge class names and
normalize into the streaming wire contract's :class:`Box` — the one detection vocabulary
HTTP and Kafka consumers share.

The class names and input resolution are read from the metadata the exporter embedded in
the graph, never configured beside it.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from mlops_cv.serving import prepost
from mlops_cv.startup import StartupError
from mlops_cv.streaming.messages import Box, ModelInfo

logger = logging.getLogger(__name__)

# The only execution provider the service accepts.
# A silent fall back to CPU would keep answering an order of magnitude slower.
CUDA_PROVIDER = "CUDAExecutionProvider"


class RuntimeSession(Protocol):
    """The slice of an onnxruntime session this module uses — and all a fake must provide."""

    def get_modelmeta(self) -> Any: ...

    def get_inputs(self) -> Sequence[Any]: ...

    def run(
        self, output_names: list[str] | None, feed: dict[str, np.ndarray], /
    ) -> Sequence[Any]: ...


def graph_spec(session: RuntimeSession) -> tuple[dict[int, str], int]:
    """Read the exporter's embedded metadata off a loaded session: ``(class names, imgsz)``."""
    metadata = session.get_modelmeta().custom_metadata_map
    try:
        raw_names, raw_imgsz = metadata["names"], metadata["imgsz"]
    except KeyError as missing:
        raise StartupError(
            f"the graph carries no {missing} metadata — it does not look like a published "
            "model graph (class names and input size are embedded at export)"
        ) from None
    try:
        # The exporter writes both as their Python repr.
        names, imgsz = ast.literal_eval(raw_names), ast.literal_eval(raw_imgsz)
    except (ValueError, SyntaxError) as exc:
        raise StartupError(f"the graph's names/imgsz metadata is unreadable: {exc}") from exc
    if not names:
        raise StartupError("the graph carries no class names")
    height, width = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    if height != width:
        raise StartupError(
            f"the graph takes a {width}x{height} input; serving letterboxes onto a square"
        )
    return {int(k): str(v) for k, v in names.items()}, int(height)


def load_session(graph: Path) -> RuntimeSession:
    """Build the CUDA-EP session for a published graph — no CPU fallback."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise StartupError(f"onnxruntime cannot load its CUDA libraries ({exc})") from exc

    session = ort.InferenceSession(str(graph), providers=[CUDA_PROVIDER])
    if CUDA_PROVIDER not in session.get_providers():
        raise StartupError(
            f"onnxruntime has no {CUDA_PROVIDER} (got {session.get_providers()}) — "
            "the service serves on the GPU only; check the CUDA runtime and the "
            "onnxruntime-gpu install"
        )
    return session


def to_boxes(
    detections: np.ndarray, names: Mapping[int, str], source_shape: tuple[int, int]
) -> list[Box]:
    """``(n, 6)`` source-pixel detections -> wire :class:`Box` es (normalized centre xywh)."""
    height, width = source_shape
    scale = np.array([width, height, width, height], dtype=np.float32)
    corners = detections[:, :4]
    centres = (corners[:, :2] + corners[:, 2:]) / 2
    sizes = corners[:, 2:] - corners[:, :2]
    xywhn = (np.concatenate([centres, sizes], axis=1) / scale).tolist()
    return [
        Box(
            cls=names.get(int(cls_id), str(int(cls_id))),
            conf=float(conf),
            xywhn=tuple(box),
        )
        for box, (conf, cls_id) in zip(xywhn, detections[:, 4:].tolist(), strict=True)
    ]


class Detector:
    """One loaded graph, answering uploads with wire boxes."""

    def __init__(self, session: RuntimeSession, model: ModelInfo) -> None:
        self.session = session
        self.model = model
        self.names, self.imgsz = graph_spec(session)
        self._input = session.get_inputs()[0].name

    @classmethod
    def load(cls, graph: Path, model: ModelInfo) -> Detector:
        """Production wiring: a CUDA-EP session over a published graph."""
        detector = cls(load_session(graph), model)
        logger.info(
            f"serving {model.name} v{model.version} from {graph.name} "
            f"at imgsz {detector.imgsz} ({len(detector.names)} classes)"
        )
        return detector

    def detect(self, image: bytes, conf_threshold: float) -> list[Box]:
        """Decode an uploaded image and answer with its detections.

        Raises ``ValueError`` for bytes that are not a decodable image — a client error,
        not a service failure.
        """
        source = prepost.load_image(image)
        tensor = prepost.preprocess(source, self.imgsz)
        output = self.session.run(None, {self._input: tensor})[0]
        source_shape = (source.shape[0], source.shape[1])
        detections = prepost.postprocess(
            output, source_shape, imgsz=self.imgsz, conf_threshold=conf_threshold
        )
        return to_boxes(detections, self.names, source_shape)
