"""On-device accuracy and latency for published NCNN variants, gated against the desktop's."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Mapping
from importlib.metadata import version as package_version
from pathlib import Path

from mlops_cv import benchmark as benchmark_mod
from mlops_cv.config import Settings, get_settings
from mlops_cv.data.subset import IMAGES_DIRNAME, stamp_split_yaml
from mlops_cv.optimize.report import _fmt
from mlops_cv.optimize.variants import NCNN, Variant, parse_artifact_tag
from mlops_cv.startup import StartupError, exits_on_startup_error
from mlops_cv.tracking import champion, client
from mlops_cv.tracking.metric_keys import (
    OPTIMIZE_PREFIX,
    PRIMARY,
    headline_metrics,
    metric_key,
    variant_device_latency_prefix,
    variant_device_prefix,
    variant_prefix,
)
from mlops_cv.tracking.resolve import download, run_id_from_uri

logger = logging.getLogger(__name__)

# The split the device measures on: the one the desktop's readings come from.
SPLIT = "test"

# Packages whose versions decide the device's numerics, tagged so a failed gate can be told apart.
VERSIONED_PACKAGES = ("ultralytics", "ncnn", "torch")


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    """CLI parser. Constructed with no heavy imports so it stays testable in CI."""
    p = argparse.ArgumentParser(
        description="Measure a model version's published NCNN variants on this device; "
        "log accuracy and latency onto the version's record run."
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
        "--device-label",
        default="pi5",
        help="namespace for this device's readings: optimize/<variant>/device/<label>",
    )
    return p


def ncnn_artifacts(tags: Mapping[str, str], names: Iterable[str] | None) -> dict[str, str]:
    """Which published NCNN artifacts to measure: ``{variant name: artifact URI}``."""
    published = {
        variant.name: uri
        for key, uri in tags.items()
        if (variant := parse_artifact_tag(key)) is not None and variant.runtime == NCNN
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


def pull_model(version: str | None, variant: str, dst: Path = Path("models")) -> Path:
    """Download one published NCNN variant of ``version`` (``None``: the champion) into ``dst``."""
    settings = get_settings()
    client.connect(settings, experiment=False)
    uri = ncnn_artifacts(champion.resolve(settings, version).tags, [variant])[variant]
    return download(uri, dst)


def record_run_id(artifact_uris: Iterable[str]) -> str:
    """The run holding the published artifacts, read out of the URIs the version tags carry.

    Not the version's own training run: the desktop logs its record to a *child* of that run, so
    following the artifact URI is what puts this device's readings beside the desktop's.
    """
    run_ids = {run_id for uri in artifact_uris if (run_id := run_id_from_uri(uri))}
    if len(run_ids) != 1:
        raise ValueError(
            f"expected one record run across the published artifacts, found {sorted(run_ids)} — "
            "re-run the optimization so every artifact comes from one ladder"
        )
    return run_ids.pop()


def accuracy_gate(device: float, desktop: float, tolerance_pt: float) -> tuple[float, bool]:
    """Δ = device - desktop mAP50-95 in points, and whether it is within ``tolerance_pt``."""
    delta_pt = 100.0 * (device - desktop)
    return delta_pt, abs(delta_pt) <= tolerance_pt


def device_report(
    model_ref: str,
    label: str,
    variants: Iterable[str],
    metrics: Mapping[str, float],
    tolerance_pt: float,
    versions: Mapping[str, str],
) -> str:
    """``edge/<label>.md``: per-variant desktop vs device accuracy, device latency, gate verdict."""
    lines = [
        f"# Edge report: {label}",
        "",
        f"Model: `{model_ref}`",
        "",
        f"| Variant | Desktop {PRIMARY} | Device {PRIMARY} | Δ pt | P50 ms | P95 ms "
        f"| Gate ±{tolerance_pt} pt |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in variants:
        desktop = metrics[metric_key(variant_prefix(v), PRIMARY)]
        device = metrics[metric_key(variant_device_prefix(v, label), PRIMARY)]
        delta_pt, within = accuracy_gate(device, desktop, tolerance_pt)
        latency = variant_device_latency_prefix(v, label)
        lines.append(
            f"| `{v}` | {_fmt(desktop)} | {_fmt(device)} | {delta_pt:+.2f} "
            f"| {_fmt(metrics.get(metric_key(latency, 'p50_ms')), 2)} "
            f"| {_fmt(metrics.get(metric_key(latency, 'p95_ms')), 2)} "
            f"| {'pass' if within else 'fail'} |"
        )
    lines += ["", "| Package | Version |", "|---|---|"]
    lines += [f"| {pkg} | {ver} |" for pkg, ver in versions.items()]
    return "\n".join(lines) + "\n"


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    args = build_parser(settings).parse_args(argv)
    label = args.device_label
    tolerance = settings.optimize.device_tolerance_pt

    subset = settings.data.subset_dir
    template = subset / settings.data.dataset_yaml.name
    images = benchmark_mod.latency_frames(subset / IMAGES_DIRNAME / SPLIT)
    if not images or not template.is_file():
        raise StartupError(
            f"no {SPLIT} split at DATA__SUBSET_DIR={subset} (needs {IMAGES_DIRNAME}/{SPLIT}, "
            f"labels/{SPLIT} and {template.name}) — rsync it from the desktop, see "
            "'Raspberry Pi 5 setup' in docs/edge.md"
        )
    data_yaml = stamp_split_yaml(template, subset, SPLIT)

    mlflow = client.connect(settings, experiment=False)

    from ultralytics import YOLO

    name = settings.mlflow.registered_model
    version = champion.resolve(settings, args.model_version)
    selected = ncnn_artifacts(version.tags, args.variants.split(",") if args.variants else None)
    if not selected:
        raise StartupError(
            f"version {version.version} advertises no NCNN artifacts — run the optimization "
            f"first (`make optimize`)"
        )

    record_run = record_run_id(selected.values())
    desktop = mlflow.get_run(record_run).data.metrics
    desktop_keys = {v: metric_key(variant_prefix(v), PRIMARY) for v in selected}
    if missing := [v for v, key in desktop_keys.items() if key not in desktop]:
        raise StartupError(
            f"run {record_run} has no desktop {PRIMARY} for {', '.join(missing)} — "
            "re-run `make optimize` for this version"
        )

    logger.info(
        f"measuring {name} v{version.version} on '{label}' -> run {record_run}: "
        f"{', '.join(selected)}"
    )
    failed: list[str] = []
    versions = {pkg: package_version(pkg) for pkg in VERSIONED_PACKAGES}
    readings = dict(desktop)
    with mlflow.start_run(run_id=record_run):
        mlflow.set_tags({f"{OPTIMIZE_PREFIX}.{label}.{pkg}": ver for pkg, ver in versions.items()})
        for variant_name, uri in selected.items():
            model = YOLO(str(download(uri)), task="detect")
            imgsz = Variant.parse(variant_name).imgsz
            prefix = variant_device_prefix(variant_name, label)
            results = model.val(
                data=str(data_yaml),
                split=SPLIT,
                imgsz=imgsz,
                batch=1,
                device="cpu",
                plots=False,
                verbose=False,
            )
            accuracy = headline_metrics(results.box, prefix=prefix)
            on_device = accuracy[metric_key(prefix, PRIMARY)]
            on_desktop = desktop[desktop_keys[variant_name]]
            delta_pt, within = accuracy_gate(on_device, on_desktop, tolerance)
            latency_prefix = variant_device_latency_prefix(variant_name, label)
            latency = benchmark_mod.benchmark_model(
                model, images, imgsz=imgsz, device="cpu", prefix=latency_prefix
            )
            logged = accuracy | latency | {metric_key(prefix, f"delta_{PRIMARY}_pt"): delta_pt}
            mlflow.log_metrics(logged)
            readings |= logged
            reading = (
                f"{variant_name}: {PRIMARY} {on_device:.4f} on {label} vs {on_desktop:.4f} on the "
                f"desktop (Δ {delta_pt:+.2f} pt)"
            )
            logger.info(f"{reading}, p50={latency[metric_key(latency_prefix, 'p50_ms')]:.1f}ms")
            if not within:
                failed.append(reading)
            del model
        report = device_report(
            f"models:/{name}/{version.version}", label, selected, readings, tolerance, versions
        )
        mlflow.log_text(report, f"edge/{label}.md")
        if failed:
            logger.error(
                f"accuracy gap beyond ±{tolerance} pt, {OPTIMIZE_PREFIX}.measured.{label} not "
                f"set: {'; '.join(failed)}"
            )
            return 1
        mlflow.set_tag(f"{OPTIMIZE_PREFIX}.measured.{label}", "true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
