"""``tracking.resolve``: ``resolve_model`` URI/branch/slug logic (mlflow faked), NCNN selection."""

from __future__ import annotations

import sys
import types

import pytest

from mlops_cv.optimize.variants import artifact_tag
from mlops_cv.tracking.resolve import (
    ncnn_artifacts,
    record_run_id,
    resolve_model,
    source_run_id,
)


class _FakeVersion:
    source = "mlflow-artifacts:/1/run/artifacts/weights/best.pt"
    run_id = "run"


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Inject fake ``mlflow`` modules and record the calls the resolver makes."""
    recorded: dict = {}

    class FakeClient:
        def get_model_version_by_alias(self, name: str, alias: str) -> _FakeVersion:
            recorded["alias"] = (name, alias)
            return _FakeVersion()

        def get_model_version(self, name: str, version: str) -> _FakeVersion:
            recorded["version"] = (name, version)
            return _FakeVersion()

    def _download(artifact_uri: str, dst_path: str | None = None) -> str:
        recorded["download_uri"] = artifact_uri
        return "/tmp/dl/best.pt"

    fake = types.ModuleType("mlflow")
    fake.MlflowClient = FakeClient  # type: ignore[attr-defined]
    artifacts = types.ModuleType("mlflow.artifacts")
    artifacts.download_artifacts = _download  # type: ignore[attr-defined]
    fake.artifacts = artifacts  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.artifacts", artifacts)
    return recorded


def test_alias_resolves_and_downloads_version_source(calls: dict) -> None:
    path, slug = resolve_model("models:/aerial-object-detector@champion")
    assert calls["alias"] == ("aerial-object-detector", "champion")
    assert calls["download_uri"] == _FakeVersion.source  # the version's own source, not a run path
    assert slug == "aerial-object-detector-champion"
    assert path.name == "best.pt"


def test_version_resolves_and_downloads_version_source(calls: dict) -> None:
    resolve_model("models:/aerial-object-detector/7")
    assert calls["version"] == ("aerial-object-detector", "7")
    assert calls["download_uri"] == _FakeVersion.source


def test_run_uri_passes_through_without_registry_lookup(calls: dict) -> None:
    _, slug = resolve_model("runs:/abc123/weights/best.pt")
    assert calls["download_uri"] == "runs:/abc123/weights/best.pt"
    assert "alias" not in calls and "version" not in calls
    assert slug == "abc123-weights-best.pt"


def test_source_run_is_the_versions_training_run_or_the_run_uris_own(calls: dict) -> None:
    """Where evaluation and optimization read the training imgsz from."""
    assert source_run_id("models:/aerial-object-detector-server@champion") == _FakeVersion.run_id
    assert source_run_id("runs:/abc123/weights/best.pt") == "abc123"
    assert source_run_id("best.pt") is None


# A model version's tags as the optimize driver writes them.
TAGS = {
    "optimize.ncnn_fp32_640": "runs:/abc/ncnn/ncnn-fp32-640_ncnn_model",
    "optimize.ncnn_fp16_640": "runs:/abc/ncnn/ncnn-fp16-640_ncnn_model",
    "optimize.ncnn_fp16_320": "runs:/abc/ncnn/ncnn-fp16-320_ncnn_model",
    "optimize.onnx_640": "runs:/abc/onnx/best_640.onnx",
    "optimize.trt_fp16_640": "runs:/abc/engines/trt-fp16-640.engine",
    "optimize.trt_fp16_640_fingerprint": "runs:/abc/engines/trt-fp16-640.fingerprint.json",
    "optimize.recommended": "should-never-exist-but-must-not-match-either",
}


def test_selects_only_ncnn_artifacts_by_default() -> None:
    """Engines and graphs are advertised under the same scheme but can't run on the device."""
    selected = ncnn_artifacts(TAGS, None)
    assert sorted(selected) == ["ncnn-fp16-320", "ncnn-fp16-640", "ncnn-fp32-640"]
    assert selected["ncnn-fp16-320"] == "runs:/abc/ncnn/ncnn-fp16-320_ncnn_model"


def test_tag_keys_round_trip_to_variant_slugs() -> None:
    """The device recovers exactly the slug the driver's tag encoder started from."""
    assert "ncnn-fp16-320" in ncnn_artifacts({artifact_tag("ncnn-fp16-320"): "u"}, None)


def test_names_filter_selects_a_subset() -> None:
    assert list(ncnn_artifacts(TAGS, ["ncnn-fp16-320"])) == ["ncnn-fp16-320"]


def test_unpublished_names_error_and_list_what_exists() -> None:
    with pytest.raises(ValueError) as exc:
        ncnn_artifacts(TAGS, ["ncnn-int8-320"])
    assert "ncnn-int8-320" in str(exc.value)
    assert "ncnn-fp16-320" in str(exc.value)


def test_no_published_artifacts_selects_nothing() -> None:
    assert ncnn_artifacts({"optimize.onnx_640": "u"}, None) == {}


def test_record_run_is_followed_from_the_artifact_uri() -> None:
    """Not the version's training run: the record lives on a child of it, and the artifact URI
    is what says which one — so the device's readings land beside the desktop's."""
    assert record_run_id(ncnn_artifacts(TAGS, None).values()) == "abc"


def test_artifacts_from_different_runs_are_refused() -> None:
    """A half-refreshed set of tags would scatter one device's readings across two ladders."""
    with pytest.raises(ValueError, match="one record run"):
        record_run_id(["runs:/abc/optimize/ncnn/a", "runs:/def/optimize/ncnn/b"])


def test_no_resolvable_run_is_refused() -> None:
    with pytest.raises(ValueError, match="one record run"):
        record_run_id(["s3://bucket/some/path"])
