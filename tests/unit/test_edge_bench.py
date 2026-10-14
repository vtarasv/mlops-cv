"""On-device harness seam: tag selection and the CLI contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlops_cv.config import Settings
from mlops_cv.config.settings import DataSettings
from mlops_cv.edge import bench
from mlops_cv.edge.bench import (
    accuracy_gate,
    build_parser,
    device_report,
    ncnn_artifacts,
    record_run_id,
)
from mlops_cv.optimize.variants import Variant, artifact_tag
from mlops_cv.tracking.metric_keys import (
    PRIMARY,
    metric_key,
    variant_device_latency_prefix,
    variant_device_prefix,
    variant_prefix,
)

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
    """The harness recovers exactly the slug the driver's tag encoder started from."""
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


def test_selected_slugs_parse_back_to_variants_with_their_resolution() -> None:
    """The predict loop reads imgsz from the parsed Variant, not from a suffix re-parse."""
    for name in ncnn_artifacts(TAGS, None):
        assert Variant.parse(name).name == name
    assert Variant.parse("ncnn-fp16-320").imgsz == 320


def test_record_run_is_followed_from_the_artifact_uri() -> None:
    """Not the version's training run: the record lives on a child of it, and the artifact URI
    is what says which one — so the device's latency lands beside the matching accuracy."""
    assert record_run_id(ncnn_artifacts(TAGS, None).values()) == "abc"


def test_artifacts_from_different_runs_are_refused() -> None:
    """A half-refreshed set of tags would scatter one device's readings across two ladders."""
    with pytest.raises(ValueError, match="one record run"):
        record_run_id(["runs:/abc/optimize/ncnn/a", "runs:/def/optimize/ncnn/b"])


def test_no_resolvable_run_is_refused() -> None:
    with pytest.raises(ValueError, match="one record run"):
        record_run_id(["s3://bucket/some/path"])


def test_parser_defaults() -> None:
    args = build_parser(Settings()).parse_args([])
    assert args.model_version is None  # champion by default
    assert args.variants is None  # every published NCNN artifact
    assert args.device_label == "pi5"


def test_parser_accepts_explicit_version_and_label() -> None:
    args = build_parser(Settings()).parse_args(
        ["--model-version", "7", "--device-label", "desktop-cpu", "--variants", "ncnn-fp16-320"]
    )
    assert args.model_version == "7"
    assert args.device_label == "desktop-cpu"
    assert args.variants == "ncnn-fp16-320"


def test_gate_measures_the_gap_in_points() -> None:
    delta, within = accuracy_gate(0.2213, 0.2218, 0.5)
    assert delta == pytest.approx(-0.05)
    assert within


@pytest.mark.parametrize("device", [0.2160, 0.2280])
def test_gate_fails_either_way_past_the_tolerance(device: float) -> None:
    _, within = accuracy_gate(device, 0.2218, 0.5)
    assert not within


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
    versions = {"2": SimpleNamespace(tags=TAGS)}
    monkeypatch.setattr(bench.client, "connect", lambda settings, experiment: None)
    monkeypatch.setattr(bench.champion, "resolve", lambda settings, version: versions[version])
    monkeypatch.setattr(bench, "download", lambda uri, dst: (uri, dst))
    got = bench.pull_model("2", "ncnn-fp16-640", tmp_path)
    assert got == ("runs:/abc/ncnn/ncnn-fp16-640_ncnn_model", tmp_path)


def test_device_report_rows_carry_both_readings_latency_and_verdict() -> None:
    def desktop(v: str) -> str:
        return metric_key(variant_prefix(v), PRIMARY)

    def device(v: str) -> str:
        return metric_key(variant_device_prefix(v, "pi5"), PRIMARY)

    def latency(v: str, p: int) -> str:
        return metric_key(variant_device_latency_prefix(v, "pi5"), f"p{p}_ms")

    metrics = {
        desktop("ncnn-fp16-640"): 0.2218,
        device("ncnn-fp16-640"): 0.2213,
        latency("ncnn-fp16-640", 50): 171.42,
        latency("ncnn-fp16-640", 95): 180.07,
        desktop("ncnn-fp16-320"): 0.1500,
        device("ncnn-fp16-320"): 0.1400,
    }
    md = device_report(
        "models:/m/2",
        "pi5",
        ["ncnn-fp16-640", "ncnn-fp16-320"],
        metrics,
        0.5,
        {"ultralytics": "8.4.0", "ncnn": "1.0.2"},
    )
    assert md == (
        "# Edge report: pi5\n"
        "\n"
        "Model: `models:/m/2`\n"
        "\n"
        "| Variant | Desktop mAP50-95 | Device mAP50-95 | Δ pt | P50 ms | P95 ms | Gate ±0.5 pt |\n"
        "|---|---|---|---|---|---|---|\n"
        "| `ncnn-fp16-640` | 0.2218 | 0.2213 | -0.05 | 171.42 | 180.07 | pass |\n"
        "| `ncnn-fp16-320` | 0.1500 | 0.1400 | -1.00 | — | — | fail |\n"
        "\n"
        "| Package | Version |\n"
        "|---|---|\n"
        "| ultralytics | 8.4.0 |\n"
        "| ncnn | 1.0.2 |\n"
    )
