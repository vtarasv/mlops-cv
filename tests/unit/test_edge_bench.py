"""On-device half of the edge gate: the CLI contract and what it leaves on the record run."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlops_cv.config import Settings
from mlops_cv.config.settings import DataSettings
from mlops_cv.edge import bench
from mlops_cv.edge.bench import build_parser
from mlops_cv.tracking import edge_record
from mlops_cv.tracking.metric_keys import metric_key, variant_device_latency_prefix

TAGS = {
    "optimize.ncnn_fp16_640": "runs:/abc/ncnn/ncnn-fp16-640_ncnn_model",
    "optimize.onnx_640": "runs:/abc/onnx/best_640.onnx",
}


def test_parser_defaults() -> None:
    args = build_parser(Settings()).parse_args([])
    assert args.model_version is None  # champion by default
    assert args.variants is None  # every published NCNN artifact
    assert args.device_label == "pi5"


def test_parser_accepts_explicit_version_and_label() -> None:
    args = build_parser(Settings()).parse_args(
        ["--model-version", "7", "--device-label", "desktop-cpu", "--variants", "ncnn-fp16-320"]
    )
    assert (args.model_version, args.device_label, args.variants) == (
        "7",
        "desktop-cpu",
        "ncnn-fp16-320",
    )


def test_main_refuses_a_missing_test_split_with_the_fix(tmp_path, monkeypatch, caplog) -> None:
    """The same exit-2-with-the-fix contract as every other entrypoint — no assert, no traceback."""
    settings = Settings(data=DataSettings(subset_dir=tmp_path))
    monkeypatch.setattr(bench, "get_settings", lambda: settings)
    with caplog.at_level("ERROR"):
        assert bench.main([]) == 2
    assert "rsync" in caplog.text


def test_pull_model_downloads_the_named_variant_of_the_resolved_version(
    monkeypatch, tmp_path
) -> None:
    versions = {("2", "edge"): SimpleNamespace(tags=TAGS, run_id="train-2")}
    monkeypatch.setattr(bench.client, "connect", lambda settings, experiment: None)
    monkeypatch.setattr(
        bench.champion, "resolve", lambda settings, version, target: versions[version, target]
    )
    monkeypatch.setattr(bench, "download", lambda uri, dst: (uri, dst))
    got = bench.pull_model("2", "ncnn-fp16-640", tmp_path)
    assert got == ("runs:/abc/ncnn/ncnn-fp16-640_ncnn_model", tmp_path)


def test_pull_model_defaults_to_the_served_variant_at_the_training_imgsz(
    monkeypatch, tmp_path
) -> None:
    """The edge champion's own training imgsz picks ncnn-fp16-<imgsz>, never a Settings value."""
    tags = TAGS | {"optimize.ncnn_fp16_1280": "runs:/abc/ncnn/ncnn-fp16-1280_ncnn_model"}
    monkeypatch.setattr(bench.client, "connect", lambda settings, experiment: None)
    monkeypatch.setattr(
        bench.champion,
        "resolve",
        lambda settings, version, target: SimpleNamespace(tags=tags, run_id="train-1"),
    )
    monkeypatch.setattr(bench.champion, "training_imgsz", lambda run_id: 1280)
    monkeypatch.setattr(bench, "download", lambda uri, dst: uri)
    assert bench.pull_model(None) == "runs:/abc/ncnn/ncnn-fp16-1280_ncnn_model"


class StubDetector:
    """An ``NcnnDetector`` that records the settings it predicts at."""

    calls: list[dict] = []

    def __init__(self, model_dir: Path, imgsz: int) -> None:
        assert imgsz == 640

    def predict(self, image: object, **kwargs: object) -> np.ndarray:
        StubDetector.calls.append(kwargs)
        return np.array([[1, 2, 3, 4, 0.5, 1]], np.float32)


def test_main_dumps_val_settings_detections_times_and_retags(
    tmp_path, monkeypatch, fake_mlflow
) -> None:
    """The dump is at val's settings, latency at predict's; stale gate and package tags go."""
    frames = tmp_path / "images" / "test"
    frames.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (frames / name).touch()
    mlflow = fake_mlflow
    mlflow.registry.add_run("abc", {"optimize.measured.pi5": "true", "optimize.pi5.torch": "2.1"})
    settings = Settings(data=DataSettings(subset_dir=tmp_path))
    monkeypatch.setattr(bench, "get_settings", lambda: settings)
    monkeypatch.setattr(bench.client, "connect", lambda settings, experiment: mlflow)
    monkeypatch.setattr(
        bench.champion, "resolve", lambda s, v, target: SimpleNamespace(version="1", tags=TAGS)
    )
    monkeypatch.setattr(bench, "download", lambda uri: tmp_path)
    monkeypatch.setattr(bench, "load_image", lambda path: path)
    monkeypatch.setattr(bench, "NcnnDetector", StubDetector)
    StubDetector.calls = []

    assert bench.main([]) == 0

    payload = mlflow.dicts[edge_record.detections_path("pi5", "ncnn-fp16-640")]
    assert payload == {name: [[1.0, 2.0, 3.0, 4.0, 0.5, 1.0]] for name in ("a.jpg", "b.jpg")}
    assert StubDetector.calls[0] == bench.VAL_SETTINGS
    assert StubDetector.calls[-1] == {}  # timed at predict defaults
    assert metric_key(variant_device_latency_prefix("ncnn-fp16-640", "pi5"), "p50_ms") in (
        mlflow.metrics
    )
    assert sorted(mlflow.deleted) == ["optimize.measured.pi5", "optimize.pi5.torch"]
    assert mlflow.tags["optimize.pi5.numpy"] == np.__version__


@pytest.mark.parametrize("pkg", bench.VERSIONED_PACKAGES)
def test_installed_versions_skip_what_is_not_installed(pkg: str, monkeypatch) -> None:
    from importlib.metadata import PackageNotFoundError

    def version(name: str) -> str:
        if name == pkg:
            raise PackageNotFoundError(name)
        return "1.0"

    monkeypatch.setattr(bench, "package_version", version)
    assert pkg not in bench.installed_versions()
    assert len(bench.installed_versions()) == len(bench.VERSIONED_PACKAGES) - 1
