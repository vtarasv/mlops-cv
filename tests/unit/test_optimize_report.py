"""Optimization report seam: the rendered artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mlops_cv.optimize.report import VariantMeasurement, write_report
from mlops_cv.optimize.variants import Variant
from mlops_cv.tracking.metric_keys import (
    PRIMARY,
    variant_latency_prefix,
    variant_prefix,
    variant_speed_prefix,
)


def _measure(
    runtime: str,
    precision: str,
    imgsz: int,
    primary: float,
    p50: float,
    vram: float | None = 86.0,
) -> VariantMeasurement:
    variant = Variant.of(runtime, precision, imgsz)
    return VariantMeasurement(
        variant=variant,
        metrics={
            f"{variant_prefix(variant.name)}/{PRIMARY}": primary,
            f"{variant_prefix(variant.name)}/mAP50": primary + 0.2,
            f"{variant_latency_prefix(variant.name)}/p50_ms": p50,
            f"{variant_latency_prefix(variant.name)}/p95_ms": p50 * 1.2,
            f"{variant_latency_prefix(variant.name)}/p99_ms": p50 * 1.5,
        },
        artifact_mb=20.0,
        vram_mb=vram,
        vram_total_mb=354.0,
        vram_scope="process",
    )


BASELINE = _measure("torch", "fp32", 640, primary=0.2000, p50=30.0)


def test_report_renders_every_variant(tmp_path: Path) -> None:
    rows = [BASELINE, _measure("trt", "fp16", 640, primary=0.199, p50=8.0)]
    md_path, csv_path = write_report(tmp_path, rows, model_ref="models:-name@champion")

    text = md_path.read_text(encoding="utf-8")
    assert "torch-fp32-640" in text
    assert "trt-fp16-640" in text
    assert "models:-name@champion" in text

    with csv_path.open(encoding="utf-8") as fh:
        parsed = list(csv.DictReader(fh))
    assert [r["variant"] for r in parsed] == ["torch-fp32-640", "trt-fp16-640"]
    assert float(parsed[0]["mAP50-95"]) == pytest.approx(0.2)
    assert float(parsed[1]["p50_ms"]) == pytest.approx(8.0)


def test_report_splits_server_and_edge_sections(tmp_path: Path) -> None:
    """GPU and CPU rungs are different deployment targets; one mixed table would invite
    comparing their latencies, which mean different things."""
    rows = [
        BASELINE,
        _measure("trt", "fp16", 640, primary=0.199, p50=8.0),
        _measure("ncnn", "fp16", 320, primary=0.115, p50=60.0, vram=None),
    ]
    md_path, csv_path = write_report(tmp_path, rows, model_ref="m")
    text = md_path.read_text(encoding="utf-8")
    server, edge = text.split("## Edge variants")
    assert "trt-fp16-640" in server
    assert "ncnn-fp16-320" not in server
    assert "ncnn-fp16-320" in edge

    with csv_path.open(encoding="utf-8") as fh:
        by_name = {r["variant"]: r for r in csv.DictReader(fh)}
    assert by_name["trt-fp16-640"]["device"] == "gpu"
    assert by_name["ncnn-fp16-320"]["device"] == "cpu"


def test_report_without_edge_rungs_has_no_edge_section(tmp_path: Path) -> None:
    md_path, _ = write_report(tmp_path, [BASELINE], model_ref="m")
    assert "## Edge variants" not in md_path.read_text(encoding="utf-8")


def test_report_shows_the_inference_only_stage(tmp_path: Path) -> None:
    """When inference falls far below end-to-end latency, pre/post dominates — the table must
    make that visible rather than leaving 'compilation barely helped' unexplained."""
    variant = Variant.of("trt", "fp16", 640)
    row = VariantMeasurement(
        variant=variant,
        metrics={
            f"{variant_prefix(variant.name)}/{PRIMARY}": 0.21,
            f"{variant_latency_prefix(variant.name)}/p50_ms": 20.0,
            f"{variant_speed_prefix(variant.name)}/inference_ms": 3.2,
            f"{variant_speed_prefix(variant.name)}/preprocess_ms": 0.4,
            f"{variant_speed_prefix(variant.name)}/postprocess_ms": 0.1,
        },
    )
    assert row.speed("inference") == 3.2
    md_path, csv_path = write_report(tmp_path, [row], model_ref="m")
    assert "Infer ms" in md_path.read_text(encoding="utf-8")
    with csv_path.open(encoding="utf-8") as fh:
        assert float(next(csv.DictReader(fh))["inference_ms"]) == pytest.approx(3.2)


def test_report_shows_added_memory_not_the_process_total(tmp_path: Path) -> None:
    """The absolute reading is dominated by a floor every variant shares, so the table reports
    the marginal cost and leaves the total to the CSV."""
    md_path, csv_path = write_report(tmp_path, [BASELINE], model_ref="m")
    assert "VRAM +MB" in md_path.read_text(encoding="utf-8")
    with csv_path.open(encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert float(row["vram_added_mb"]) == pytest.approx(86.0)
    assert float(row["vram_process_total_mb"]) == pytest.approx(354.0)


def test_engine_fingerprints_render_when_present(tmp_path: Path) -> None:
    """Published engines are host-bound; the fingerprint is what tells a consumer whether one
    is usable on its hardware."""
    variant = Variant.of("trt", "fp16", 640)
    row = VariantMeasurement(
        variant=variant,
        metrics={f"{variant_prefix(variant.name)}/{PRIMARY}": 0.21},
        fingerprint={"sha256": "abc123", "gpu": "RTX 5070", "trt": "10.16.1.11"},
    )
    text = write_report(tmp_path, [row], model_ref="m")[0].read_text(encoding="utf-8")
    assert "abc123" in text
    assert "RTX 5070" in text
