"""Serving-variant registry seam."""

from __future__ import annotations

import pytest

from mlops_cv.optimize.variants import (
    FP16,
    FP32,
    NCNN,
    ONNX_ORT,
    OPSET,
    TORCH,
    TRT,
    Variant,
    engine_export_kwargs,
    graph_export_kwargs,
    ladder,
    ncnn_export_kwargs,
    select,
)


def test_ladder_golden_slugs_and_order() -> None:
    """Each step changes exactly one deployment decision; the report renders in this order."""
    assert [v.name for v in ladder(640, 320)] == [
        "torch-fp32-640",
        "onnx-ort-fp32-640",
        "trt-fp16-640",
        "ncnn-fp32-640",
        "ncnn-fp16-640",
        "ncnn-fp16-320",
    ]


def test_ladder_follows_configured_resolutions() -> None:
    names = [v.name for v in ladder(512, 256)]
    assert names[0] == "torch-fp32-512"
    assert names[-1] == "ncnn-fp16-256"


def test_baseline_is_the_first_rung() -> None:
    rungs = ladder(640, 320)
    assert rungs[0].runtime == TORCH
    assert rungs[0].precision == "fp32"
    assert rungs[0].is_baseline


def test_slug_is_runtime_precision_resolution() -> None:
    assert Variant.of(TRT, "int8", 320).name == "trt-int8-320"


def test_deployment_roles_never_appear_in_slugs() -> None:
    """Roles are interpretations that belong in prose; identities outlive them."""
    for variant in ladder(640, 320):
        assert "server" not in variant.name
        assert "edge" not in variant.name


def test_ncnn_rungs_decompose_the_edge_candidates_loss() -> None:
    """fp32-640 anchors, fp16-640 isolates precision, fp16-320 isolates resolution."""
    rungs = {v.name: v for v in ladder(640, 320)}
    assert rungs["ncnn-fp16-640"].imgsz == rungs["ncnn-fp32-640"].imgsz
    assert rungs["ncnn-fp16-640"].precision == rungs["ncnn-fp16-320"].precision


def test_graph_variants_need_the_exported_onnx_graph() -> None:
    """NCNN traces the framework model directly (PNNX) — it never touches the ONNX graph."""
    rungs = {v.name: v for v in ladder(640, 320)}
    assert rungs["torch-fp32-640"].needs_graph is False
    assert rungs["onnx-ort-fp32-640"].needs_graph is True
    assert rungs["trt-fp16-640"].needs_graph is True
    assert rungs["ncnn-fp16-320"].needs_graph is False


def test_only_trt_variants_are_compiled() -> None:
    """NCNN artifacts are portable directories, not host-bound binaries."""
    rungs = {v.name: v for v in ladder(640, 320)}
    assert rungs["trt-fp16-640"].is_compiled is True
    assert rungs["onnx-ort-fp32-640"].is_compiled is False
    assert rungs["ncnn-fp16-320"].is_compiled is False


def test_ncnn_variants_measure_on_the_cpu() -> None:
    """CPU rungs never occupy the GPU; their desktop latency is a relative reading only."""
    rungs = {v.name: v for v in ladder(640, 320)}
    assert rungs["ncnn-fp16-320"].measure_device("0") == "cpu"
    assert rungs["trt-fp16-640"].measure_device("0") == "0"
    assert rungs["torch-fp32-640"].measure_device("cuda:1") == "cuda:1"


def test_graph_export_kwargs_are_static_single_image() -> None:
    kwargs = graph_export_kwargs(640)
    assert kwargs["format"] == "onnx"
    assert kwargs["imgsz"] == 640
    assert kwargs["batch"] == 1
    assert kwargs["dynamic"] is False  # a dynamic profile would keep 2x shapes viable for nothing
    assert kwargs["simplify"] is True
    assert kwargs["opset"] == OPSET


def test_opset_is_pinned() -> None:
    """The export library subtracts 2 on a CUDA device; pinned so a torch bump can't move it."""
    assert OPSET == 18


def test_ncnn_export_kwargs_golden() -> None:
    """Only the args the NCNN exporter declares — no opset, no dynamic, no simplify."""
    assert ncnn_export_kwargs(Variant.of(NCNN, FP16, 320)) == {
        "format": "ncnn",
        "imgsz": 320,
        "half": True,
        "batch": 1,
    }
    assert ncnn_export_kwargs(Variant.of(NCNN, FP32, 640))["half"] is False


def test_engine_export_kwargs_fp16() -> None:
    variant = Variant.of(TRT, FP16, 640)
    kwargs = engine_export_kwargs(variant, workspace_gb=4)
    assert kwargs["format"] == "engine"
    assert kwargs["half"] is True
    assert kwargs["imgsz"] == 640
    assert kwargs["batch"] == 1
    assert kwargs["dynamic"] is False
    assert kwargs["workspace"] == 4


def test_engine_export_pins_the_opset_too() -> None:
    """The compiler re-exports the graph internally from the same opset argument — unpinned,
    the engine rung would follow whatever a future framework version picks."""
    assert engine_export_kwargs(Variant.of(TRT, FP16, 640), workspace_gb=4)["opset"] == OPSET


def test_precision_constants_are_golden() -> None:
    assert (FP32, FP16) == ("fp32", "fp16")


def test_select_filters_to_named_variants_keeping_ladder_order() -> None:
    chosen = select(ladder(640, 320), ["ncnn-fp16-320", "torch-fp32-640"])
    assert [v.name for v in chosen] == ["torch-fp32-640", "ncnn-fp16-320"]


def test_select_without_names_returns_the_whole_ladder() -> None:
    assert len(select(ladder(640, 320), None)) == 6


def test_select_rejects_unknown_names_and_lists_the_valid_ones() -> None:
    with pytest.raises(ValueError) as exc:
        select(ladder(640, 320), ["trt-fp16-999"])
    assert "trt-fp16-999" in str(exc.value)
    assert "torch-fp32-640" in str(exc.value)


def test_runtime_constants_are_golden() -> None:
    assert (TORCH, ONNX_ORT, TRT, NCNN) == ("torch", "onnx-ort", "trt", "ncnn")
