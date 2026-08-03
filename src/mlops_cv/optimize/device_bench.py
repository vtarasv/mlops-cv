"""On-device latency for published NCNN variants: the edge half of the Optimization report.

Runs on the deployment device (e.g., Raspberry Pi 5, but any CPU host works): pulls each NCNN
artifact the model version's tags advertise, times a predict loop over a small local image set,
and logs the readings onto the **same record run** the desktop measured accuracy on (found by
following the artifact URI) — under a device-labeled namespace
(``optimize/<variant>/latency/<device-label>/…``).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

from mlops_cv.config import Settings, get_settings
from mlops_cv.optimize import benchmark as benchmark_mod
from mlops_cv.optimize.variants import NCNN
from mlops_cv.tracking import client
from mlops_cv.tracking.metric_keys import OPTIMIZE_PREFIX, variant_device_latency_prefix

logger = logging.getLogger(__name__)

# Model-version tag prefix that advertises an NCNN artifact (``optimize.ncnn_fp16_320`` ...).
NCNN_TAG_PREFIX = f"{OPTIMIZE_PREFIX}.{NCNN}_"


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    """CLI parser. Constructed with no heavy imports so it stays testable in CI."""
    p = argparse.ArgumentParser(
        description="Benchmark a model version's published NCNN variants on this device; "
        "log latency onto the version's tracking run."
    )
    p.add_argument(
        "--model-version",
        default=None,
        help=f"registered {settings.mlflow.registered_model} version (default: the champion alias)",
    )
    p.add_argument(
        "--variants",
        default=None,
        help="comma-separated NCNN variant names to measure (default: every published one)",
    )
    p.add_argument(
        "--images",
        type=Path,
        default=Path("bench-images"),
        help="local directory of frames the predict loop cycles through",
    )
    p.add_argument(
        "--device-label",
        default="pi5",
        help="namespace for this device's readings: optimize/<variant>/latency/<label>/...",
    )
    return p


def ncnn_artifacts(tags: Mapping[str, str], names: Iterable[str] | None) -> dict[str, str]:
    """Which published NCNN artifacts to measure: ``{variant name: artifact URI}``."""
    published = {
        key.removeprefix(f"{OPTIMIZE_PREFIX}.").replace("_", "-"): uri
        for key, uri in tags.items()
        if key.startswith(NCNN_TAG_PREFIX)
    }
    if names is None:
        return published
    wanted = set(names)
    if unknown := wanted - set(published):
        raise ValueError(
            f"not published on this version: {', '.join(sorted(unknown))}. "
            f"Published: {', '.join(sorted(published)) or 'none'}"
        )
    return {name: uri for name, uri in published.items() if name in wanted}


def variant_imgsz(variant_name: str) -> int:
    """The input resolution a variant slug encodes (its final ``-<imgsz>`` segment)."""
    return int(variant_name.rsplit("-", 1)[1])


def record_run_id(artifact_uris: Iterable[str]) -> str:
    """The run holding the published artifacts, read out of the URIs the version tags carry.

    Not the version's own training run: the desktop logs its record to a *child* of that run, so
    following the artifact URI is what puts this device's latency on the same run as the
    accuracy it belongs beside.
    """
    run_ids = {
        uri.removeprefix("runs:/").split("/", 1)[0]
        for uri in artifact_uris
        if uri.startswith("runs:/")
    }
    if len(run_ids) != 1:
        raise ValueError(
            f"expected one record run across the published artifacts, found {sorted(run_ids)} — "
            "re-run the optimization so every artifact comes from one ladder"
        )
    return run_ids.pop()


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    args = build_parser(settings).parse_args(argv)

    images = [str(p) for p in sorted(args.images.glob("*.jpg"))]
    if not images:
        logger.error(f"no .jpg frames in {args.images} — copy a few once, any frames work")
        return 2

    client.configure()  # export MLFLOW_TRACKING_URI before importing mlflow

    import mlflow
    from mlflow import MlflowClient
    from ultralytics import YOLO

    mlflow.set_tracking_uri(client.tracking_uri())
    registry = MlflowClient()
    name = settings.mlflow.registered_model
    version = (
        registry.get_model_version(name, args.model_version)
        if args.model_version
        else registry.get_model_version_by_alias(name, settings.mlflow.champion_alias)
    )
    selected = ncnn_artifacts(version.tags, args.variants.split(",") if args.variants else None)
    if not selected:
        logger.error(f"version {version.version} advertises no NCNN artifacts — run optimize first")
        return 2

    record_run = record_run_id(selected.values())
    logger.info(
        f"benchmarking {name} v{version.version} on '{args.device_label}' "
        f"({len(images)} frames) -> run {record_run}: {', '.join(selected)}"
    )

    from mlflow.artifacts import download_artifacts

    with mlflow.start_run(run_id=record_run):
        for variant_name, uri in selected.items():
            local = Path(download_artifacts(artifact_uri=uri))
            model = YOLO(str(local), task="detect")
            imgsz = variant_imgsz(variant_name)
            timings = benchmark_mod.time_calls(
                lambda src, m=model, s=imgsz: m.predict(src, imgsz=s, device="cpu", verbose=False),
                images,
            )
            prefix = variant_device_latency_prefix(variant_name, args.device_label)
            metrics = benchmark_mod.latency_metrics(timings, prefix=prefix)
            mlflow.log_metrics(metrics)
            logger.info(
                f"{variant_name}: p50={metrics[f'{prefix}/p50_ms']:.1f}ms on {args.device_label}"
            )
            del model
        mlflow.set_tag(f"{OPTIMIZE_PREFIX}.measured.{args.device_label}", "true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
