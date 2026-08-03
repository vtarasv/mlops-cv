"""The Optimization report: the Serving-variant tables, one per deployment target."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mlops_cv.optimize.variants import NCNN, Variant
from mlops_cv.tracking.metric_keys import (
    PRIMARY,
    metric_key,
    variant_latency_prefix,
    variant_prefix,
    variant_speed_prefix,
)


@dataclass(frozen=True)
class VariantMeasurement:
    variant: Variant
    metrics: Mapping[str, float]
    artifact_mb: float | None = None
    # GPU memory this variant added.
    vram_mb: float | None = None
    # Whole-process GPU memory while this variant ran (shared floor included).
    vram_total_mb: float | None = None
    vram_scope: str | None = None
    build_s: float | None = None
    # A compiled engine's host binding: content hash, build args, toolchain, GPU.
    fingerprint: Mapping[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.variant.name

    def reading(self, name: str) -> float | None:
        """One accuracy reading by metric name, e.g. ``mAP50-95``."""
        return self.metrics.get(metric_key(variant_prefix(self.name), name))

    def latency(self, percentile: int) -> float | None:
        return self.metrics.get(metric_key(variant_latency_prefix(self.name), f"p{percentile}_ms"))

    def speed(self, stage: str) -> float | None:
        """One stage of the library's breakdown: ``preprocess``/``inference``/``postprocess``."""
        return self.metrics.get(metric_key(variant_speed_prefix(self.name), f"{stage}_ms"))

    @property
    def primary(self) -> float | None:
        return self.reading(PRIMARY)

    @property
    def is_edge(self) -> bool:
        """Whether this rung targets the edge device (CPU/NCNN) rather than the server GPU."""
        return self.variant.runtime == NCNN


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _table_md(measurements: Sequence[VariantMeasurement]) -> str:
    header = (
        "| Variant | imgsz | mAP50-95 | mAP50 | P50 ms | P95 ms | P99 ms | Infer ms "
        "| Artifact MB | VRAM +MB |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|---|---|"]
    for m in measurements:
        lines.append(
            f"| `{m.name}` | {m.variant.imgsz} | {_fmt(m.primary)} "
            f"| {_fmt(m.reading('mAP50'))} "
            f"| {_fmt(m.latency(50), 2)} | {_fmt(m.latency(95), 2)} | {_fmt(m.latency(99), 2)} "
            f"| {_fmt(m.speed('inference'), 2)} "
            f"| {_fmt(m.artifact_mb, 1)} | {_fmt(m.vram_mb, 0)} |"
        )
    return "\n".join(lines)


def _fingerprints_md(measurements: Sequence[VariantMeasurement]) -> str:
    described = [m for m in measurements if m.fingerprint]
    if not described:
        return ""
    keys = sorted({k for m in described for k in m.fingerprint})
    lines = ["| Variant | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for m in described:
        lines.append(
            f"| `{m.name}` | " + " | ".join(str(m.fingerprint.get(k, "—")) for k in keys) + " |"
        )
    return "\n".join(lines)


def _csv_row(m: VariantMeasurement) -> dict[str, object]:
    return {
        "variant": m.name,
        "runtime": m.variant.runtime,
        "precision": m.variant.precision,
        "imgsz": m.variant.imgsz,
        "device": "cpu" if m.is_edge else "gpu",
        "mAP50-95": m.primary,
        "mAP50": m.reading("mAP50"),
        "precision_metric": m.reading("precision"),
        "recall": m.reading("recall"),
        "p50_ms": m.latency(50),
        "p95_ms": m.latency(95),
        "p99_ms": m.latency(99),
        "preprocess_ms": m.speed("preprocess"),
        "inference_ms": m.speed("inference"),
        "postprocess_ms": m.speed("postprocess"),
        "artifact_mb": m.artifact_mb,
        "vram_added_mb": m.vram_mb,
        "vram_process_total_mb": m.vram_total_mb,
        "vram_scope": m.vram_scope,
        "build_s": m.build_s,
        **{f"fingerprint_{k}": v for k, v in m.fingerprint.items()},
    }


def write_report(
    out_dir: str | Path,
    measurements: Sequence[VariantMeasurement],
    *,
    model_ref: str,
) -> tuple[Path, Path]:
    """Write ``optimization.md`` + ``variants.csv``; return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    server = [m for m in measurements if not m.is_edge]
    edge = [m for m in measurements if m.is_edge]

    sections = [
        "# Optimization report",
        "",
        f"Model: `{model_ref}`",
        "",
    ]
    if server:
        sections += [
            "## Server variants (host GPU)",
            "",
            _table_md(server),
            "",
        ]
    if edge:
        sections += [
            "## Edge variants (NCNN on host CPU)",
            "",
            _table_md(edge),
            "",
        ]

    fingerprints = _fingerprints_md(measurements)
    if fingerprints:
        sections += [
            "## Compiled engines",
            "",
            fingerprints,
            "",
        ]

    md_path = out / "optimization.md"
    md_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")

    rows = [_csv_row(m) for m in measurements]
    fieldnames: list[str] = []
    for row in rows:
        fieldnames += [k for k in row if k not in fieldnames]
    csv_path = out / "variants.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return md_path, csv_path
