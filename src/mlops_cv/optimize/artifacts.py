"""Producing a Serving variant's artifact: the portable graph, NCNN models, compiled engines."""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mlops_cv.optimize.variants import (
    Variant,
    engine_export_kwargs,
    graph_export_kwargs,
    ncnn_export_kwargs,
)

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

# What the NCNN runtime actually loads: the graph, its weights, and the class/imgsz metadata.
NCNN_KEEP_SUFFIXES = (".param", ".bin")
NCNN_KEEP_NAMES = ("metadata.yaml",)


def _load_for_export(weights: str | Path) -> YOLO:
    """Load the checkpoint for export"""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    args = getattr(model.model, "args", None)
    if isinstance(args, dict) and args.get("data"):
        args["data"] = Path(args["data"]).name
    return model


def prune_ncnn_dir(model_dir: Path) -> list[str]:
    """Drop everything the NCNN runtime doesn't load; return the names removed."""
    removed = []
    for entry in sorted(model_dir.iterdir()):
        if entry.is_file() and (
            entry.suffix in NCNN_KEEP_SUFFIXES or entry.name in NCNN_KEEP_NAMES
        ):
            continue
        shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        removed.append(entry.name)
    return removed


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def size_mb(path: str | Path) -> float:
    """Artifact size in MB; a directory artifact (NCNN) counts every file inside."""
    target = Path(path)
    if target.is_dir():
        return sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e6
    return target.stat().st_size / 1e6


def export_graph(weights: str | Path, imgsz: int, out_dir: str | Path) -> tuple[Path, float]:
    """Export the portable graph at one resolution; return its path and the build seconds."""
    kwargs = graph_export_kwargs(imgsz)
    logger.info(f"exporting portable graph at imgsz={imgsz} (opset {kwargs['opset']})")
    start = time.perf_counter()
    produced = _load_for_export(weights).export(**kwargs)
    elapsed = time.perf_counter() - start

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"{Path(produced).stem}_{imgsz}.onnx"
    shutil.move(str(produced), destination)
    logger.info(f"graph {destination.name}: {size_mb(destination):.1f} MB in {elapsed:.1f}s")
    return destination, elapsed


def export_ncnn(weights: str | Path, variant: Variant, out_dir: str | Path) -> tuple[Path, float]:
    """Export ``variant``'s NCNN model (a portable directory); return its path and build seconds."""
    kwargs = ncnn_export_kwargs(variant)
    logger.info(f"exporting {variant.name} (half={kwargs['half']})")
    start = time.perf_counter()
    produced = _load_for_export(weights).export(**kwargs)
    elapsed = time.perf_counter() - start

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"{variant.name}_ncnn_model"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(produced), destination)
    if removed := prune_ncnn_dir(destination):
        logger.info(f"pruned from {destination.name}: {', '.join(removed)}")
    logger.info(f"ncnn {destination.name}: {size_mb(destination):.1f} MB in {elapsed:.1f}s")
    return destination, elapsed


def build_engine(
    weights: str | Path,
    variant: Variant,
    out_dir: str | Path,
    *,
    workspace_gb: int,
) -> tuple[Path, float]:
    """Compile ``variant``'s engine into ``out_dir``; return its path and the build seconds.

    The export library re-exports the graph internally before compiling.
    """
    kwargs = engine_export_kwargs(variant, workspace_gb=workspace_gb)
    logger.info(f"compiling {variant.name}")
    start = time.perf_counter()
    produced = _load_for_export(weights).export(**kwargs)
    elapsed = time.perf_counter() - start

    out = Path(out_dir) / f"{variant.name}.engine"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), out)
    logger.info(f"engine {out.name}: {size_mb(out):.1f} MB in {elapsed:.1f}s")
    return out, elapsed


def engine_fingerprint(path: str | Path, variant: Variant, *, workspace_gb: int) -> dict[str, str]:
    """What a consumer needs to decide whether a published engine is valid for it.

    An engine only runs on the GPU architecture + compiler version that built it; this record
    travels next to the binary so the check happens before a load fails.
    """
    import tensorrt as trt
    import torch

    major, minor = torch.cuda.get_device_capability(0)
    return {
        "sha256": sha256(path)[:16],
        "size_mb": f"{size_mb(path):.1f}",
        "trt": trt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "sm": f"{major}{minor}",
        "precision": variant.precision,
        "imgsz": str(variant.imgsz),
        "workspace_gb": str(workspace_gb),
    }
