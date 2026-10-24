"""What the service serves: the champion's published graph, resolved once at startup."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlops_cv.config import Settings
from mlops_cv.optimize.variants import SERVER, graph_tag
from mlops_cv.startup import StartupError
from mlops_cv.streaming.messages import ModelInfo
from mlops_cv.tracking import champion
from mlops_cv.tracking.resolve import download

if TYPE_CHECKING:
    from mlflow.entities.model_registry import ModelVersion

    from mlops_cv.serving.runtime import Detector


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


def resolve_graph(settings: Settings, *, registry: Any | None = None) -> tuple[Path, ModelInfo]:
    """Server champion -> its graph at its training imgsz, and the version that published it."""
    version = champion.resolve(settings, target=SERVER, registry=registry)
    local = download(graph_uri(version, champion.training_imgsz(version.run_id, registry=registry)))
    return local, ModelInfo(name=settings.mlflow.registered_model(SERVER), version=version.version)


def champion_detector(settings: Settings) -> Detector:
    """Production wiring: the champion's graph, downloaded and loaded on the GPU."""
    from mlops_cv.serving.runtime import Detector

    return Detector.load(*resolve_graph(settings))
