"""On-device harness seam: tag selection and the CLI contract."""

from __future__ import annotations

import pytest

from mlops_cv.config import Settings
from mlops_cv.optimize.device_bench import (
    build_parser,
    ncnn_artifacts,
    record_run_id,
)
from mlops_cv.optimize.variants import Variant, artifact_tag

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
    assert args.images.name == "bench-images"


def test_parser_accepts_explicit_version_and_label() -> None:
    args = build_parser(Settings()).parse_args(
        ["--model-version", "7", "--device-label", "desktop-cpu", "--variants", "ncnn-fp16-320"]
    )
    assert args.model_version == "7"
    assert args.device_label == "desktop-cpu"
    assert args.variants == "ncnn-fp16-320"


def test_main_refuses_an_empty_image_directory_with_the_fix(tmp_path, caplog) -> None:
    """The same exit-2-with-the-fix contract as every other entrypoint — no assert, no traceback."""
    from mlops_cv.optimize import device_bench

    with caplog.at_level("ERROR"):
        assert device_bench.main(["--images", str(tmp_path)]) == 2
    assert "copy a few" in caplog.text
