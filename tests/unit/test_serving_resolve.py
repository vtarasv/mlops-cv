"""Champion -> published-graph resolution: the service's three fail-fast startup paths.

No live MLflow: the registry lookup is faked and handed model-version objects shaped like
the ones the optimization driver tags.
"""

from __future__ import annotations

import pytest

from mlops_cv.config import Settings
from mlops_cv.optimize.variants import graph_tag
from mlops_cv.serving import resolve as serving_resolve
from mlops_cv.serving.errors import StartupError
from mlops_cv.serving.resolve import graph_uri, resolve_graph


class FakeVersion:
    """A registered model version as the registry returns it."""

    def __init__(self, tags: dict[str, str], version: str = "7") -> None:
        self.tags = tags
        self.version = version
        self.run_id = "train-run"


PUBLISHED = {
    "optimize.onnx_640": "runs:/record-run/onnx/best_640.onnx",
    "optimize.trt_fp16_640": "runs:/record-run/engines/trt-fp16-640.engine",
    "optimize.ncnn_fp16_320": "runs:/record-run/ncnn/ncnn-fp16-320_ncnn_model",
}


def test_graph_uri_reads_the_published_onnx_tag() -> None:
    assert graph_uri(FakeVersion(PUBLISHED), 640) == "runs:/record-run/onnx/best_640.onnx"  # type: ignore


def test_graph_uri_addresses_the_requested_resolution() -> None:
    """The tag grammar is the owned encoder — a second resolution has its own tag."""
    version = FakeVersion({graph_tag(320): "runs:/record-run/onnx/best_320.onnx"})
    assert graph_uri(version, 320).endswith("best_320.onnx")  # type: ignore
    with pytest.raises(StartupError):
        graph_uri(version, 640)  # type: ignore


def test_missing_graph_tag_names_the_command_that_publishes_it() -> None:
    """A champion that was never optimized: the engine/NCNN tags don't stand in for it."""
    without_graph = {k: v for k, v in PUBLISHED.items() if k != graph_tag(640)}
    with pytest.raises(StartupError) as exc:
        graph_uri(FakeVersion(without_graph), 640)  # type: ignore
    message = str(exc.value)
    assert "optimize.onnx_640" in message  # which tag is missing
    assert "make optimize" in message  # and how to produce it
    assert "version 7" in message  # on which version


def test_resolve_graph_downloads_the_tagged_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    graph = tmp_path / "best_640.onnx"
    graph.write_bytes(b"onnx")
    monkeypatch.setattr(serving_resolve, "model_version", lambda ref, name: FakeVersion(PUBLISHED))
    monkeypatch.setattr(serving_resolve, "download_graph", lambda uri: graph)

    resolved = resolve_graph(Settings())
    assert resolved.path == graph
    assert resolved.model.name == "aerial-object-detector"
    assert resolved.model.version == "7"


def test_resolve_graph_asks_the_registry_for_the_champion_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Rollout is restart-to-pick-up: the alias is what the service resolves, once."""
    asked: dict[str, str] = {}

    def lookup(ref: str, name: str) -> FakeVersion:
        asked["ref"], asked["name"] = ref, name
        return FakeVersion(PUBLISHED)

    monkeypatch.setattr(serving_resolve, "model_version", lookup)
    monkeypatch.setattr(serving_resolve, "download_graph", lambda uri: tmp_path)

    settings = Settings()
    resolve_graph(settings)
    assert asked["ref"] == "models:/aerial-object-detector@champion"
    assert asked["name"] == settings.mlflow.registered_model


def test_no_champion_alias_fails_fast_with_the_promotion_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordinary state of a fresh install — reported plainly, not as a traceback."""
    from mlflow.exceptions import MlflowException

    def missing(ref: str, name: str) -> None:
        raise MlflowException("Registered model alias champion not found.")

    monkeypatch.setattr(serving_resolve, "model_version", missing)
    with pytest.raises(StartupError) as exc:
        resolve_graph(Settings())
    message = str(exc.value)
    assert "champion" in message
    assert "aerial-object-detector" in message
    assert "make train" in message


def test_unresolvable_champion_fails_fast_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ref that maps to no registered version is the same startup failure, not a None."""
    monkeypatch.setattr(serving_resolve, "model_version", lambda ref, name: None)
    with pytest.raises(StartupError, match="champion"):
        resolve_graph(Settings())
