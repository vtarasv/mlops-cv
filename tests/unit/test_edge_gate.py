"""Desktop half of the edge gate: scoring the device's dumps, the Δ gate and the report."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlops_cv.config import Settings
from mlops_cv.config.settings import DataSettings
from mlops_cv.optimize import edge_gate
from mlops_cv.optimize.edge_gate import accuracy_gate, build_parser, device_report
from mlops_cv.tracking import edge_record
from mlops_cv.tracking.metric_keys import (
    PRIMARY,
    metric_key,
    variant_device_latency_prefix,
    variant_device_prefix,
    variant_prefix,
)

VARIANT = "ncnn-fp16-640"
TAGS = {"optimize.ncnn_fp16_640": f"runs:/abc/ncnn/{VARIANT}_ncnn_model"}
DESKTOP = metric_key(variant_prefix(VARIANT), PRIMARY)
DEVICE = metric_key(variant_device_prefix(VARIANT, "pi5"), PRIMARY)


def test_parser_defaults() -> None:
    args = build_parser(Settings()).parse_args([])
    assert (args.model_version, args.device_label) == (None, "pi5")


def test_gate_measures_the_gap_in_points() -> None:
    delta, within = accuracy_gate(0.2213, 0.2218, 0.5)
    assert delta == pytest.approx(-0.05)
    assert within


@pytest.mark.parametrize("device", [0.2160, 0.2280])
def test_gate_fails_either_way_past_the_tolerance(device: float) -> None:
    _, within = accuracy_gate(device, 0.2218, 0.5)
    assert not within


@pytest.fixture
def gate(tmp_path, monkeypatch, fake_mlflow):
    """``edge_gate.main`` over a subset in ``tmp_path`` and a record run ``abc``."""
    settings = Settings(data=DataSettings(subset_dir=tmp_path))
    (tmp_path / "images" / "test").mkdir(parents=True)
    monkeypatch.setattr(edge_gate, "get_settings", lambda: settings)
    monkeypatch.setattr(edge_gate.client, "connect", lambda settings, experiment: fake_mlflow)
    monkeypatch.setattr(edge_gate.client, "registry", lambda settings: fake_mlflow.registry)
    monkeypatch.setattr(
        edge_gate.champion,
        "resolve",
        lambda s, v, target: SimpleNamespace(version="1", tags=TAGS) if target == "edge" else None,
    )
    monkeypatch.setattr("mlflow.artifacts.load_dict", lambda uri: {"a.jpg": []})

    def run(device_map: float, desktop: dict[str, float], dumped: bool = True) -> int:
        fake_mlflow.registry.add_run("abc", {"optimize.pi5.ncnn": "1.0"}, desktop)
        if dumped:
            fake_mlflow.registry.artifacts.add(("abc", edge_record.detections_path("pi5", VARIANT)))
        box = SimpleNamespace(mp=0.5, mr=0.4, map50=0.45, map=device_map)
        monkeypatch.setattr(
            "mlops_cv.evaluation.score_detections.score",
            lambda dets, yaml, split: SimpleNamespace(box=box),
        )
        return edge_gate.main([])

    return run


def test_a_pass_logs_the_device_readings_and_sets_measured(gate, fake_mlflow) -> None:
    assert gate(0.2207, {DESKTOP: 0.2218}) == 0
    assert fake_mlflow.metrics[DEVICE] == 0.2207
    delta = metric_key(variant_device_prefix(VARIANT, "pi5"), f"delta_{PRIMARY}_pt")
    assert fake_mlflow.metrics[delta] == pytest.approx(-0.11)
    assert fake_mlflow.tags == {edge_record.measured_tag("pi5"): "true"}
    assert "| ncnn | 1.0 |" in fake_mlflow.texts[edge_record.report_path("pi5")]


def test_a_gap_past_the_tolerance_exits_1_without_measured(gate, fake_mlflow) -> None:
    assert gate(0.2100, {DESKTOP: 0.2218}) == 1
    assert fake_mlflow.tags == {}
    assert "| fail |" in fake_mlflow.texts[edge_record.report_path("pi5")]


def test_a_missing_desktop_reading_names_make_optimize(gate, caplog) -> None:
    with caplog.at_level("ERROR"):
        assert gate(0.2207, {}) == 2
    assert "make optimize" in caplog.text


def test_a_missing_dump_names_the_device_bench(gate, caplog) -> None:
    with caplog.at_level("ERROR"):
        assert gate(0.2207, {DESKTOP: 0.2218}, dumped=False) == 2
    assert "python -m mlops_cv.edge.bench" in caplog.text


def test_a_missing_subset_names_make_ingest(gate, tmp_path, caplog) -> None:
    (tmp_path / "images" / "test").rmdir()
    with caplog.at_level("ERROR"):
        assert gate(0.2207, {DESKTOP: 0.2218}) == 2
    assert "make ingest" in caplog.text


def test_device_report_rows_carry_both_readings_latency_and_verdict() -> None:
    def desktop(v: str) -> str:
        return metric_key(variant_prefix(v), PRIMARY)

    def device(v: str) -> str:
        return metric_key(variant_device_prefix(v, "pi5"), PRIMARY)

    def latency(v: str, p: int) -> str:
        return metric_key(variant_device_latency_prefix(v, "pi5"), f"p{p}_ms")

    metrics = {
        desktop("ncnn-fp16-640"): 0.2218,
        device("ncnn-fp16-640"): 0.2213,
        latency("ncnn-fp16-640", 50): 171.42,
        latency("ncnn-fp16-640", 95): 180.07,
        desktop("ncnn-fp16-320"): 0.1500,
        device("ncnn-fp16-320"): 0.1400,
    }
    md = device_report(
        "models:/m/2",
        "pi5",
        ["ncnn-fp16-640", "ncnn-fp16-320"],
        metrics,
        0.5,
        {"ncnn": "1.0.2", "numpy": "2.4.6"},
    )
    assert md == (
        "# Edge report: pi5\n"
        "\n"
        "Model: `models:/m/2`\n"
        "\n"
        "| Variant | Desktop mAP50-95 | Device mAP50-95 | Δ pt | P50 ms | P95 ms | Gate ±0.5 pt |\n"
        "|---|---|---|---|---|---|---|\n"
        "| `ncnn-fp16-640` | 0.2218 | 0.2213 | -0.05 | 171.42 | 180.07 | pass |\n"
        "| `ncnn-fp16-320` | 0.1500 | 0.1400 | -1.00 | — | — | fail |\n"
        "\n"
        "| Package | Version |\n"
        "|---|---|\n"
        "| ncnn | 1.0.2 |\n"
        "| numpy | 2.4.6 |\n"
    )
