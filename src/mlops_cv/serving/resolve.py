"""What the service serves: the champion's published graph, resolved once at startup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlops_cv.config import Settings
from mlops_cv.optimize.variants import graph_tag
from mlops_cv.startup import StartupError
from mlops_cv.streaming.messages import ModelInfo
from mlops_cv.tracking import champion
from mlops_cv.tracking.resolve import download

if TYPE_CHECKING:
    from mlflow.entities.model_registry import ModelVersion

    from mlops_cv.serving.runtime import Detector


@dataclass(frozen=True)
class ResolvedGraph:
    """A local ONNX graph plus the identity of the model version that published it."""

    path: Path
    model: ModelInfo


def graph_uri(version: ModelVersion, imgsz: int) -> str:
    """The published-graph artifact URI ``version`` advertises at ``imgsz``."""
    key = graph_tag(imgsz)
    uri = version.tags.get(key)
    if not uri:
        raise StartupError(
            f"model version {version.version} carries no '{key}' tag — the champion has no "
            f"published graph at imgsz {imgsz}. Run `make optimize` to build and publish it."
        )
    return uri


def resolve_graph(settings: Settings, *, registry: Any | None = None) -> ResolvedGraph:
    """Champion alias -> the local graph to load and the model."""
    version = champion.resolve(settings, registry=registry)
    local = download(graph_uri(version, settings.optimize.server_imgsz))
    return ResolvedGraph(
        path=local,
        model=ModelInfo(name=settings.mlflow.registered_model, version=version.version),
    )


def champion_detector(settings: Settings) -> Detector:
    """Production wiring: the champion's graph, downloaded and loaded on the GPU."""
    from mlops_cv.serving.runtime import Detector

    resolved = resolve_graph(settings)
    return Detector.load(resolved.path, resolved.model)
