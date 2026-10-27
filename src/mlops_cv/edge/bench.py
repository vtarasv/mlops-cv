"""On-device half of the edge gate: latency and a detections dump per published NCNN variant.

Torch-free: the variants run on `edge.runtime`, and the desktop scores the dumps
(`python -m mlops_cv.optimize.edge_gate`).
"""

from __future__ import annotations

import argparse
import contextlib
import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from mlops_cv import benchmark as benchmark_mod
from mlops_cv.config import Settings, get_settings
from mlops_cv.data.subset import IMAGES_DIRNAME
from mlops_cv.edge.runtime import NcnnDetector
from mlops_cv.optimize.variants import EDGE, Variant, served_variant
from mlops_cv.serving.prepost import load_image
from mlops_cv.startup import StartupError, exits_on_startup_error
from mlops_cv.tracking import champion, client, edge_record
from mlops_cv.tracking.metric_keys import metric_key, variant_device_latency_prefix
from mlops_cv.tracking.resolve import download, ncnn_artifacts, record_run_id

logger = logging.getLogger(__name__)

# The split the device dumps: the one the desktop's readings come from.
SPLIT = "test"

# ultralytics val's NMS settings. A single-label dump reads 0.30 pt lower.
VAL_SETTINGS = {"conf": 0.001, "iou": 0.7, "max_det": 300, "multi_label": True}

# Packages whose versions move the device's detections, tagged so a failed gate can be explained.
VERSIONED_PACKAGES = ("ncnn", "numpy", "opencv-python", "opencv-python-headless")


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    """CLI parser. Constructed with no heavy imports so it stays testable in CI."""
    p = argparse.ArgumentParser(
        description="Time a model version's published NCNN variants on this device and dump "
        "their test-split detections onto the version's record run, for the desktop to score."
    )
    p.add_argument(
        "--model-version",
        default=None,
        help=f"registered {settings.mlflow.registered_model(EDGE)} version "
        "(default: the champion alias)",
    )
    p.add_argument(
        "--variants",
        default=None,
        help="comma-separated NCNN variant names to measure (default: every published one)",
    )
    p.add_argument(
        "--device-label",
        default="pi5",
        help="namespace for this device's readings: optimize/<variant>/device/<label>",
    )
    return p


def installed_versions() -> dict[str, str]:
    """The installed ``VERSIONED_PACKAGES``; an OpenCV build installs one of its two wheels."""
    versions = {}
    for pkg in VERSIONED_PACKAGES:
        with contextlib.suppress(PackageNotFoundError):
            versions[pkg] = package_version(pkg)
    return versions


def pull_model(version: str | None, variant: str | None = None, dst: Path = Path("models")) -> Path:
    """Download one published NCNN variant of an edge ``version`` (``None``: the champion) into
    ``dst``; ``variant`` defaults to the served one at the version's training imgsz."""
    settings = get_settings()
    client.connect(settings, experiment=False)
    entity = champion.resolve(settings, version, target=EDGE)
    if variant is None:
        variant = served_variant(EDGE, champion.training_imgsz(entity.run_id)).name
    return download(ncnn_artifacts(entity.tags, [variant])[variant], dst)


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    args = build_parser(settings).parse_args(argv)
    label = args.device_label

    images_dir = settings.data.subset_dir / IMAGES_DIRNAME / SPLIT
    frames = sorted(images_dir.glob("*.jpg"))
    if not frames:
        raise StartupError(
            f"no {SPLIT} frames in {images_dir} (DATA__SUBSET_DIR) — rsync the split from the "
            "desktop, see 'Set up the Pi' in docs/edge.md"
        )

    mlflow = client.connect(settings, experiment=False)
    version = champion.resolve(settings, args.model_version, target=EDGE)
    selected = ncnn_artifacts(version.tags, args.variants.split(",") if args.variants else None)
    if not selected:
        raise StartupError(
            f"version {version.version} advertises no NCNN artifacts — run the optimization "
            f"first (`make optimize`)"
        )
    record_run = record_run_id(selected.values())
    logger.info(
        f"measuring {settings.mlflow.registered_model(EDGE)} v{version.version} on '{label}' "
        f"-> run {record_run}: {', '.join(selected)}"
    )

    tags = mlflow.get_run(record_run).data.tags
    # New readings are ungated, and a package this device no longer has must not linger.
    stale = {edge_record.package_tag(label, p) for p in edge_record.package_versions(tags, label)}
    stale |= {edge_record.measured_tag(label)} & tags.keys()
    with mlflow.start_run(run_id=record_run):
        for tag in stale:
            mlflow.delete_tag(tag)
        mlflow.set_tags(
            {edge_record.package_tag(label, pkg): v for pkg, v in installed_versions().items()}
        )
        for variant_name, uri in selected.items():
            detector = NcnnDetector(download(uri), Variant.parse(variant_name).imgsz)
            dump = {p.name: detector.predict(load_image(p), **VAL_SETTINGS) for p in frames}
            mlflow.log_dict(
                edge_record.to_payload(dump), edge_record.detections_path(label, variant_name)
            )
            prefix = variant_device_latency_prefix(variant_name, label)
            timings = benchmark_mod.time_calls(
                lambda path, detector=detector: detector.predict(load_image(path)),
                benchmark_mod.latency_frames(images_dir),
            )
            latency = benchmark_mod.latency_metrics(timings, prefix=prefix)
            mlflow.log_metrics(latency)
            logger.info(
                f"{variant_name}: {sum(map(len, dump.values()))} detections over {len(dump)} "
                f"frames, p50={latency[metric_key(prefix, 'p50_ms')]:.1f}ms"
            )
    model_flag = f" --model-version {args.model_version}" if args.model_version else ""
    logger.info(
        "score and gate on the desktop: python -m mlops_cv.optimize.edge_gate"
        f"{model_flag} --device-label {label}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
