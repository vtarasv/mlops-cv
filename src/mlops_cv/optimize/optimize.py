"""Produce a model version's Serving variants, measure each, and log the Optimization report.

Resolves a model the same way evaluation and streaming do, walks the variant ladder — baseline,
portable runtime, compiled engine, NCNN edge models — measuring detection quality and latency
for each on the same split, and records the result on a **child of the model version's tracking
run**: namespaced metrics, the report artifacts, and **every produced artifact** (ONNX graph,
engines with their fingerprints, NCNN directories), each addressable through a model-version tag.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlops_cv import benchmark as benchmark_mod
from mlops_cv.config import Settings, get_settings
from mlops_cv.optimize import artifacts
from mlops_cv.optimize import report as report_mod
from mlops_cv.optimize.report import VariantMeasurement
from mlops_cv.optimize.variants import (
    NCNN,
    ONNX_ORT,
    TORCH,
    TRT,
    Variant,
    artifact_tag,
    fingerprint_tag,
    graph_tag,
    ladder,
    select,
)
from mlops_cv.tracking import client
from mlops_cv.tracking.metric_keys import (
    headline_metrics,
    variant_latency_prefix,
    variant_prefix,
    variant_speed_prefix,
)
from mlops_cv.tracking.resolve import model_version, resolve_model

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Images cycled through the latency benchmark. Every variant sees the same ones.
LATENCY_IMAGES = 16

# Stages of the export library's own timing breakdown, recorded next to end-to-end latency.
SPEED_STAGES = ("preprocess", "inference", "postprocess")

# Name of the child run a version's variant record lands on.
RECORD_RUN_NAME = "optimize"

# Tag whose presence marks a run as an optimization record (set on every run this driver opens).
RECORD_TAG = "optimize.model"


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build and benchmark a model's serving variants; log the optimization report."
    )
    p.add_argument(
        "--model",
        default=None,
        help="MLflow model URI (default: the champion alias). The pipeline passes the promoted "
        "version's own runs:/ weights URI",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="the model version's training run — the record lands on a child of it "
        "(default: a standalone optimize-* run)",
    )
    p.add_argument(
        "--variants",
        default=None,
        help="comma-separated variant names to measure (default: the whole ladder)",
    )
    p.add_argument(
        "--data", type=Path, default=None, help="dataset YAML (default: the subset YAML)"
    )
    p.add_argument("--split", default="test", help="dataset split to measure on")
    p.add_argument("--device", default=settings.training.device)
    return p


def model_uri(args: argparse.Namespace, settings: Settings) -> str:
    """Which model to optimize: an explicit URI, else the champion."""
    return str(args.model) if args.model else client.champion_uri(settings)


def record_tags(
    model_ref: str, split: str, version: str | None, source_run: str | None
) -> dict[str, str]:
    """Provenance for one optimization record: what was measured, and what it resolved to."""
    tags = {RECORD_TAG: model_ref, "optimize.split": split}
    if version:
        tags["optimize.version"] = version
    if source_run:
        tags["optimize.source_run"] = source_run
    return tags


def existing_record_run(client: Any, parent_run_id: str, experiment_id: str) -> str | None:
    """An earlier optimization record."""
    children = client.search_runs(
        [experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
    )
    return next((r.info.run_id for r in children if RECORD_TAG in r.data.tags), None)


def _measure_variant(
    model: YOLO,
    variant: Variant,
    *,
    data_yaml: Path,
    split: str,
    device: str,
    images: list[str],
) -> dict[str, float]:
    """Detection readings + latency for one variant."""
    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=variant.imgsz,
        device=device,
        # Batch 1 for every variant: engines are compiled single-image.
        batch=1,
        verbose=False,
    )
    metrics = headline_metrics(results.box, prefix=variant_prefix(variant.name))
    # The library's own per-stage split, alongside the end-to-end timing.
    speed = getattr(results, "speed", None) or {}
    metrics |= {
        f"{variant_speed_prefix(variant.name)}/{stage}_ms": float(speed[stage])
        for stage in SPEED_STAGES
        if stage in speed
    }
    metrics |= benchmark_mod.benchmark_model(
        model,
        images,
        imgsz=variant.imgsz,
        device=device,
        prefix=variant_latency_prefix(variant.name),
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    args = build_parser(settings).parse_args(argv)

    mlflow = client.connect(settings)

    import torch
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException
    from ultralytics import YOLO

    if torch.cuda.is_available():
        # Touch CUDA before the first reading: until a context exists this process isn't listed
        # as a compute process, so the memory probe would fall back to a device-wide figure and
        # the first variant's delta would be measured against a different quantity than the rest.
        torch.zeros(1, device="cuda")

    opt = settings.optimize
    data_yaml = (args.data or settings.data.subset_dir / settings.data.dataset_yaml.name).resolve()
    chosen = select(
        ladder(opt.server_imgsz, opt.edge_imgsz),
        args.variants.split(",") if args.variants else None,
    )
    uri = model_uri(args, settings)
    try:
        weights, model_ref = resolve_model(uri)
    except MlflowException as exc:
        # Nothing has been promoted yet.
        if args.model:
            logger.error(f"cannot resolve --model {uri}: {exc}")
        else:
            logger.error(
                f"no '{settings.mlflow.champion_alias}' alias on "
                f"'{settings.mlflow.registered_model}' — train and promote a model first, or "
                f"name one explicitly with --model <uri>"
            )
        return 2

    registered = model_version(uri, settings.mlflow.registered_model)
    version = registered.version if registered else None
    source_run = registered.run_id if registered else None
    if version is None:
        logger.warning(
            f"{uri} maps to no registered '{settings.mlflow.registered_model}' version — "
            "artifacts will publish on the run but no model-version tags will be set"
        )
    images = [
        str(p)
        for p in sorted((settings.data.subset_dir / "images" / args.split).glob("*.jpg"))[
            :LATENCY_IMAGES
        ]
    ]

    logger.info(
        f"optimizing {model_ref} (version {version or 'unregistered'}, "
        f"trained on run {source_run or 'unknown'}) on "
        f"{data_yaml.name}[{args.split}]: {', '.join(v.name for v in chosen)}"
    )

    started = time.perf_counter()
    work_dir = Path(tempfile.mkdtemp(prefix="optimize-"))
    with contextlib.ExitStack() as stack:
        if args.run_id:
            # The record goes to a CHILD of the model version's run, never onto it.
            stack.enter_context(mlflow.start_run(run_id=args.run_id))
            parent = mlflow.active_run().info  # type: ignore
            previous = existing_record_run(MlflowClient(), parent.run_id, parent.experiment_id)
            stack.enter_context(
                mlflow.start_run(run_id=previous)
                if previous
                else mlflow.start_run(nested=True, run_name=RECORD_RUN_NAME)
            )
        else:
            stack.enter_context(mlflow.start_run(run_name=f"optimize-{model_ref}"))

        mlflow.autolog(disable=True)
        mlflow.set_tags(record_tags(model_ref, args.split, version, source_run))
        run_id = mlflow.active_run().info.run_id  # type: ignore

        def publish(local: Path, artifact_path: str, tag_key: str) -> None:
            """Log one artifact (file or directory) to the run; address it from the version."""
            if local.is_dir():
                mlflow.log_artifacts(str(local), artifact_path=artifact_path)
            else:
                mlflow.log_artifact(str(local), artifact_path=artifact_path)
                artifact_path = f"{artifact_path}/{local.name}"
            if version:
                mlflow.set_model_version_tag(
                    settings.mlflow.registered_model,
                    version,
                    tag_key,
                    f"runs:/{run_id}/{artifact_path}",
                )

        graphs: dict[int, Path] = {}
        measurements: list[VariantMeasurement] = []

        for variant in chosen:
            build_s: float | None = None
            fingerprint: dict[str, str] = {}

            if variant.needs_graph and variant.imgsz not in graphs:
                graphs[variant.imgsz], build_s = artifacts.export_graph(
                    weights, variant.imgsz, work_dir
                )

            if variant.runtime == TORCH:
                loadable: Path = Path(weights)
            elif variant.runtime == ONNX_ORT:
                loadable = graphs[variant.imgsz]
            elif variant.runtime == TRT:
                loadable, build_s = artifacts.build_engine(
                    weights,
                    variant,
                    work_dir / "engines",
                    workspace_gb=opt.workspace_gb,
                )
                fingerprint = artifacts.engine_fingerprint(
                    loadable, variant, workspace_gb=opt.workspace_gb
                )
                # The fingerprint travels next to the binary: an engine only runs on the GPU
                # architecture + compiler that built it, and the check must be possible before
                # a load fails.
                sidecar = loadable.with_suffix(".fingerprint.json")
                sidecar.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
                publish(sidecar, "engines", fingerprint_tag(variant.name))
                publish(loadable, "engines", artifact_tag(variant.name))
            elif variant.runtime == NCNN:
                loadable, build_s = artifacts.export_ncnn(weights, variant, work_dir / "ncnn")
                publish(loadable, f"ncnn/{loadable.name}", artifact_tag(variant.name))
            else:
                raise ValueError(f"unknown runtime: {variant.runtime}")

            device = variant.measure_device(args.device)
            on_gpu = device != "cpu"
            before = benchmark_mod.read_gpu_memory() if on_gpu else None
            model = YOLO(str(loadable), task="detect")
            metrics = _measure_variant(
                model,
                variant,
                data_yaml=data_yaml,
                split=args.split,
                device=device,
                images=images,
            )
            mlflow.log_metrics(metrics)
            memory = benchmark_mod.read_gpu_memory() if on_gpu else None
            added_mb = benchmark_mod.memory_delta(before, memory)
            del model
            if on_gpu:
                torch.cuda.empty_cache()

            measurements.append(
                VariantMeasurement(
                    variant=variant,
                    metrics=metrics,
                    artifact_mb=artifacts.size_mb(loadable),
                    vram_mb=added_mb,
                    vram_total_mb=memory.used_mb if memory else None,
                    vram_scope=memory.scope if memory else None,
                    build_s=build_s,
                    fingerprint=fingerprint,
                )
            )
            latest = measurements[-1]
            primary = "—" if latest.primary is None else f"{latest.primary:.4f}"
            p50 = "—" if latest.latency(50) is None else f"{latest.latency(50):.2f}ms"
            logger.info(f"{variant.name}: mAP50-95={primary} p50={p50} ({device})")

        for imgsz, graph in sorted(graphs.items()):
            publish(graph, "onnx", graph_tag(imgsz))

        md_path, csv_path = report_mod.write_report(work_dir, measurements, model_ref=model_ref)
        mlflow.log_artifact(str(md_path))
        mlflow.log_artifact(str(csv_path))

        elapsed_s = time.perf_counter() - started
        mlflow.log_metric("optimize/elapsed_s", elapsed_s)
        logger.info(f"{len(measurements)} variants in {elapsed_s / 60:.1f} min -> run {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
