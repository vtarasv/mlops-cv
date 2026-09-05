"""Champion -> published-graph resolution: the service's fail-fast startup paths.

No live MLflow: the shared in-memory registry stands in, holding model versions shaped like the
ones the optimization driver tags.
"""

from __future__ import annotations

import pytest

from mlops_cv.config import Settings
from mlops_cv.optimize.variants import graph_tag
from mlops_cv.serving import resolve as serving_resolve
from mlops_cv.serving.resolve import graph_uri, resolve_graph
from mlops_cv.startup import StartupError

PUBLISHED = {
    "optimize.onnx_640": "runs:/record-run/onnx/best_640.onnx",
    "optimize.trt_fp16_640": "runs:/record-run/engines/trt-fp16-640.engine",
    "optimize.ncnn_fp16_320": "runs:/record-run/ncnn/ncnn-fp16-320_ncnn_model",
}


def test_graph_uri_reads_the_published_onnx_tag(registry) -> None:
    version = registry.promote(version="7", tags=PUBLISHED)
    assert graph_uri(version, 640) == "runs:/record-run/onnx/best_640.onnx"  # type: ignore


def test_graph_uri_addresses_the_requested_resolution(registry) -> None:
    """The tag grammar is the owned encoder — a second resolution has its own tag."""
    version = registry.promote(tags={graph_tag(320): "runs:/record-run/onnx/best_320.onnx"})
    assert graph_uri(version, 320).endswith("best_320.onnx")  # type: ignore
    with pytest.raises(StartupError):
        graph_uri(version, 640)  # type: ignore


def test_missing_graph_tag_names_the_command_that_publishes_it(registry) -> None:
    """A champion that was never optimized: the engine/NCNN tags don't stand in for it."""
    without_graph = {k: v for k, v in PUBLISHED.items() if k != graph_tag(640)}
    version = registry.promote(version="7", tags=without_graph)
    with pytest.raises(StartupError) as exc:
        graph_uri(version, 640)  # type: ignore
    message = str(exc.value)
    assert "optimize.onnx_640" in message  # which tag is missing
    assert "make optimize" in message  # and how to produce it
    assert "version 7" in message  # on which version


def test_resolve_graph_downloads_the_tagged_artifact(
    registry, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    graph = tmp_path / "best_640.onnx"
    graph.write_bytes(b"onnx")
    registry.promote(version="7", tags=PUBLISHED)
    asked: list[str] = []

    def download(uri: str):
        asked.append(uri)
        return graph

    monkeypatch.setattr(serving_resolve, "download", download)

    path, model = resolve_graph(Settings(), registry=registry)
    assert path == graph
    assert model.name == "aerial-object-detector"
    assert model.version == "7"
    assert asked == ["runs:/record-run/onnx/best_640.onnx"]


def test_resolve_graph_asks_the_registry_for_the_champion_alias(
    registry, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Rollout is restart-to-pick-up: the alias is what the service resolves, once."""
    registry.promote(version="7", tags=PUBLISHED)
    monkeypatch.setattr(serving_resolve, "download", lambda uri: tmp_path)

    settings = Settings()
    resolve_graph(settings, registry=registry)
    assert registry.asked == [("alias", settings.mlflow.registered_model, "champion")]


def test_no_champion_alias_fails_fast_with_the_promotion_hint(registry) -> None:
    """The ordinary state of a fresh install — reported plainly, not as a traceback."""
    with pytest.raises(StartupError) as exc:
        resolve_graph(Settings(), registry=registry)
    message = str(exc.value)
    assert "champion" in message
    assert "aerial-object-detector" in message
    assert "make train" in message
