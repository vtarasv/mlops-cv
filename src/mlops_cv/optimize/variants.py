"""The Serving-variant registry: the single author of variant identity, the model-version tag
grammar that addresses published variant artifacts, and export arguments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from mlops_cv.tracking.metric_keys import OPTIMIZE_PREFIX

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

    @classmethod
    def parse(cls, name: str) -> Variant:
        """Recover a Variant from its slug (the inverse of :meth:`of`).

        The slug is mechanical — ``<runtime>-<precision>-<imgsz>`` where only the runtime may
        itself contain hyphens — so the two fixed-format fields split off the right.
        """
        parts = name.rsplit("-", 2)
        if len(parts) != 3 or not all(parts) or not parts[2].isdigit():
            raise ValueError(f"not a variant slug: {name!r}")
        runtime, precision, imgsz = parts
        return cls.of(runtime, precision, int(imgsz))

    @property
    def needs_graph(self) -> bool:
        return self.runtime in (ONNX_ORT, TRT)

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


ARTIFACT_TAG_PREFIX = f"{OPTIMIZE_PREFIX}."


def artifact_tag(variant_name: str) -> str:
    """The model-version tag key addressing one published variant artifact."""
    return ARTIFACT_TAG_PREFIX + variant_name.replace("-", "_")


def parse_artifact_tag(key: str) -> Variant | None:
    """The Variant a model-version tag key addresses, or ``None`` for any other tag."""
    if not key.startswith(ARTIFACT_TAG_PREFIX):
        return None
    name = key.removeprefix(ARTIFACT_TAG_PREFIX).replace("_", "-")
    try:
        return Variant.parse(name)
    except ValueError:
        return None


def fingerprint_tag(variant_name: str) -> str:
    """The tag key addressing a compiled engine's fingerprint sidecar."""
    return artifact_tag(variant_name) + "_fingerprint"


def graph_tag(imgsz: int) -> str:
    """The tag key addressing the portable ONNX graph at one resolution."""
    return f"{ARTIFACT_TAG_PREFIX}onnx_{imgsz}"


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
