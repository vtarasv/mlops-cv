"""The Serving-variant registry: the single author of variant identity and export arguments.

Pure stdlib: imported by CI and by the orchestrator's DAG parse.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

TORCH = "torch"
ONNX_ORT = "onnx-ort"
TRT = "trt"
NCNN = "ncnn"

FP32 = "fp32"
FP16 = "fp16"

# Graph opset. Back-compatible.
# The export library derives this from the torch version + CUDA device.
OPSET = 18

# Export policy shared by every produced artifact: one image, fixed shape, simplified graph.
_STATIC_EXPORT = {"batch": 1, "dynamic": False, "simplify": True}


@dataclass(frozen=True)
class Variant:
    """One candidate way of running a model: runtime + precision + input resolution."""

    name: str
    runtime: str
    precision: str
    imgsz: int

    @classmethod
    def of(cls, runtime: str, precision: str, imgsz: int) -> Variant:
        return cls(
            name=f"{runtime}-{precision}-{imgsz}", runtime=runtime, precision=precision, imgsz=imgsz
        )

    @property
    def is_baseline(self) -> bool:
        return self.runtime == TORCH and self.precision == FP32

    @property
    def needs_graph(self) -> bool:
        return self.runtime in (ONNX_ORT, TRT)

    @property
    def is_compiled(self) -> bool:
        """Whether this variant is an ahead-of-time compiled engine, bound to one GPU + compiler."""
        return self.runtime == TRT

    def measure_device(self, cuda_device: str) -> str:
        """The device this variant is measured on: NCNN targets the CPU, everything else the GPU."""
        return "cpu" if self.runtime == NCNN else cuda_device


def ladder(server_imgsz: int, edge_imgsz: int) -> tuple[Variant, ...]:
    """The measured variants."""
    return (
        Variant.of(TORCH, FP32, server_imgsz),
        Variant.of(ONNX_ORT, FP32, server_imgsz),
        Variant.of(TRT, FP16, server_imgsz),
        Variant.of(NCNN, FP32, server_imgsz),
        Variant.of(NCNN, FP16, server_imgsz),
        Variant.of(NCNN, FP16, edge_imgsz),
    )


def select(variants: Sequence[Variant], names: Iterable[str] | None) -> tuple[Variant, ...]:
    """Filter to ``names`` (ladder order preserved); ``None`` keeps the whole ladder."""
    if names is None:
        return tuple(variants)
    wanted = set(names)
    known = {v.name for v in variants}
    if unknown := wanted - known:
        raise ValueError(
            f"unknown variant(s): {', '.join(sorted(unknown))}. Valid: {', '.join(sorted(known))}"
        )
    return tuple(v for v in variants if v.name in wanted)


def graph_export_kwargs(imgsz: int) -> dict[str, object]:
    """Export arguments for the portable graph at one resolution."""
    return {"format": "onnx", "imgsz": imgsz, "opset": OPSET, **_STATIC_EXPORT}


def ncnn_export_kwargs(variant: Variant) -> dict[str, object]:
    """Export arguments for an NCNN model (the CPU/edge format).

    NCNN's exporter accepts only ``batch`` and ``half`` beyond the essentials — no opset (the
    trace goes through PNNX, not ONNX), no dynamic shapes.
    """
    return {
        "format": "ncnn",
        "imgsz": variant.imgsz,
        "half": variant.precision == FP16,
        "batch": 1,
    }


def engine_export_kwargs(variant: Variant, *, workspace_gb: int) -> dict[str, object]:
    """Export arguments for a compiled engine."""
    return {
        "format": "engine",
        "imgsz": variant.imgsz,
        "opset": OPSET,
        "half": variant.precision == FP16,
        "workspace": workspace_gb,
        **_STATIC_EXPORT,
    }
