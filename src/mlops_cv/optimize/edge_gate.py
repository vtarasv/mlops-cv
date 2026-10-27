"""Desktop half of the edge gate: score each device dump with ultralytics, gate it, record it."""

from __future__ import annotations

import argparse
import logging
import posixpath
from collections.abc import Iterable, Mapping

from mlops_cv.config import Settings, get_settings
from mlops_cv.data.subset import IMAGES_DIRNAME, stamp_split_yaml
from mlops_cv.optimize.report import _fmt
from mlops_cv.optimize.variants import EDGE
from mlops_cv.startup import StartupError, exits_on_startup_error
from mlops_cv.tracking import champion, client, edge_record
from mlops_cv.tracking.metric_keys import (
    PRIMARY,
    headline_metrics,
    metric_key,
    variant_device_latency_prefix,
    variant_device_prefix,
    variant_prefix,
)
from mlops_cv.tracking.resolve import ncnn_artifacts, record_run_id

logger = logging.getLogger(__name__)

# The split the device dumps and the desktop's readings come from.
SPLIT = "test"


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score a device's NCNN detections dumps on this machine's test split and gate "
        "them against the desktop's readings on the version's record run."
    )
    p.add_argument(
        "--model-version",
        default=None,
        help=f"registered {settings.mlflow.registered_model(EDGE)} version "
        "(default: the champion alias)",
    )
    p.add_argument("--device-label", default="pi5", help="the label the device bench logged under")
    return p


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
    if not (subset / IMAGES_DIRNAME / SPLIT).is_dir():
        raise StartupError(
            f"no {SPLIT} split under {subset} (DATA__SUBSET_DIR) — run `make ingest`"
        )
    data_yaml = stamp_split_yaml(settings.data.dataset_yaml, subset, SPLIT)

    mlflow = client.connect(settings, experiment=False)
    from mlflow.artifacts import load_dict

    from mlops_cv.evaluation.score_detections import score

    version = champion.resolve(settings, args.model_version, target=EDGE)
    selected = ncnn_artifacts(version.tags, None)
    if not selected:
        raise StartupError(
            f"version {version.version} advertises no NCNN artifacts — run `make optimize` first"
        )
    record_run = record_run_id(selected.values())
    run = mlflow.get_run(record_run)
    desktop = run.data.metrics
    if missing := [v for v in selected if metric_key(variant_prefix(v), PRIMARY) not in desktop]:
        raise StartupError(
            f"run {record_run} has no desktop {PRIMARY} for {', '.join(missing)} — "
            "re-run `make optimize` for this version"
        )
    reg = client.registry(settings)
    dumps = {v: edge_record.detections_path(label, v) for v in selected}
    if missing := [
        v
        for v, path in dumps.items()
        if path not in {a.path for a in reg.list_artifacts(record_run, posixpath.dirname(path))}
    ]:
        raise StartupError(
            f"run {record_run} has no '{label}' detections for {', '.join(missing)} — run "
            f"`python -m mlops_cv.edge.bench --device-label {label}` on the device first"
        )

    failed: list[str] = []
    readings = dict(desktop)
    with mlflow.start_run(run_id=record_run):
        for variant_name, path in dumps.items():
            dump = load_dict(f"runs:/{record_run}/{path}")
            metrics = score(edge_record.from_payload(dump), data_yaml, SPLIT)
            prefix = variant_device_prefix(variant_name, label)
            accuracy = headline_metrics(metrics.box, prefix=prefix)
            on_device = accuracy[metric_key(prefix, PRIMARY)]
            on_desktop = desktop[metric_key(variant_prefix(variant_name), PRIMARY)]
            delta_pt, within = accuracy_gate(on_device, on_desktop, tolerance)
            logged = accuracy | {metric_key(prefix, f"delta_{PRIMARY}_pt"): delta_pt}
            mlflow.log_metrics(logged)
            readings |= logged
            reading = (
                f"{variant_name}: {PRIMARY} {on_device:.4f} on {label} vs {on_desktop:.4f} on the "
                f"desktop (Δ {delta_pt:+.2f} pt)"
            )
            logger.info(reading)
            if not within:
                failed.append(reading)
        report = device_report(
            f"models:/{settings.mlflow.registered_model(EDGE)}/{version.version}",
            label,
            selected,
            readings,
            tolerance,
            edge_record.package_versions(run.data.tags, label),
        )
        mlflow.log_text(report, edge_record.report_path(label))
        if failed:
            logger.error(
                f"accuracy gap beyond ±{tolerance} pt, {edge_record.measured_tag(label)} not "
                f"set: {'; '.join(failed)}"
            )
            return 1
        mlflow.set_tag(edge_record.measured_tag(label), "true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
