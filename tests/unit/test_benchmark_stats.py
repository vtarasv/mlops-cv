"""Benchmark statistics seam: percentile/mean summarisation, the timing loop's contract, and
the per-process-vs-device memory-scope choice. Pure — no device, no wall-clock assertions."""

from __future__ import annotations

import pytest

from mlops_cv.optimize.benchmark import (
    DEFAULT_PERCENTILES,
    GpuMemory,
    latency_metrics,
    memory_delta,
    percentiles,
    select_used_mb,
    time_calls,
)


def test_percentiles_median() -> None:
    assert percentiles([10, 20, 30, 40, 50], (50,))[50.0] == pytest.approx(30.0)


def test_percentiles_linear_interpolation() -> None:
    out = percentiles(list(range(1, 101)), (50, 95))
    assert out[50.0] == pytest.approx(50.5)
    assert out[95.0] == pytest.approx(95.05)


def test_percentiles_empty_returns_zeros() -> None:
    assert percentiles([], (50, 95)) == {50.0: 0.0, 95.0: 0.0}


def test_latency_metrics_reports_percentiles_mean_and_count() -> None:
    out = latency_metrics(list(range(1, 101)))
    assert out["latency/p50_ms"] == pytest.approx(50.5)
    assert out["latency/p95_ms"] == pytest.approx(95.05)
    assert out["latency/p99_ms"] == pytest.approx(99.01)
    assert out["latency/mean_ms"] == pytest.approx(50.5)
    assert out["latency/n"] == 100.0


def test_latency_metrics_single_sample() -> None:
    out = latency_metrics([7.5])
    assert out["latency/p50_ms"] == pytest.approx(7.5)
    assert out["latency/p99_ms"] == pytest.approx(7.5)
    assert out["latency/mean_ms"] == pytest.approx(7.5)
    assert out["latency/n"] == 1.0


def test_latency_metrics_fewer_samples_than_percentiles() -> None:
    """Two samples still yield every requested percentile (interpolated, not dropped)."""
    out = latency_metrics([10.0, 20.0])
    assert {"latency/p50_ms", "latency/p95_ms", "latency/p99_ms"} <= set(out)
    assert out["latency/p50_ms"] == pytest.approx(15.0)
    assert out["latency/n"] == 2.0


def test_latency_metrics_empty_is_all_zeros() -> None:
    out = latency_metrics([])
    assert out == {
        "latency/p50_ms": 0.0,
        "latency/p95_ms": 0.0,
        "latency/p99_ms": 0.0,
        "latency/mean_ms": 0.0,
        "latency/n": 0.0,
    }


def test_latency_metrics_prefix_is_applied() -> None:
    out = latency_metrics([1.0], prefix="optimize/trt-fp16-640/latency")
    assert "optimize/trt-fp16-640/latency/p50_ms" in out
    assert not any(k.startswith("latency/") for k in out)


def test_latency_metrics_percentile_subset_emits_only_those_keys() -> None:
    """The evaluation harness asks for P50/P95 only — no P99 key may appear for it."""
    out = latency_metrics([1.0, 2.0, 3.0], ps=(50.0, 95.0))
    assert set(out) == {
        "latency/p50_ms",
        "latency/p95_ms",
        "latency/mean_ms",
        "latency/n",
    }


def test_default_percentiles_are_p50_p95_p99() -> None:
    assert DEFAULT_PERCENTILES == (50.0, 95.0, 99.0)


def test_time_calls_discards_warmup_and_returns_one_timing_per_iteration() -> None:
    seen: list[str] = []
    timings = time_calls(seen.append, ["a", "b"], warmup=3, iterations=5)
    assert len(timings) == 5
    assert len(seen) == 8  # warm-up calls happen, their timings are discarded
    assert all(t >= 0.0 for t in timings)


def test_time_calls_cycles_through_the_items() -> None:
    seen: list[str] = []
    time_calls(seen.append, ["a", "b", "c"], warmup=0, iterations=7)
    assert seen == ["a", "b", "c", "a", "b", "c", "a"]


def test_time_calls_without_items_does_nothing() -> None:
    calls: list[object] = []
    assert time_calls(calls.append, [], warmup=3, iterations=5) == []
    assert calls == []


def test_select_used_mb_prefers_this_process() -> None:
    reading = select_used_mb({42: 1500.0, 7: 900.0}, pid=42, device_used_mb=3000.0)
    assert reading == GpuMemory(used_mb=1500.0, scope="process")


def test_select_used_mb_falls_back_to_device_when_pid_is_absent() -> None:
    """Container PID namespaces make the vendor library report host pids we can't match."""
    reading = select_used_mb({7: 900.0}, pid=42, device_used_mb=3000.0)
    assert reading == GpuMemory(used_mb=3000.0, scope="device")


def test_select_used_mb_falls_back_when_no_processes_are_reported() -> None:
    assert select_used_mb({}, pid=42, device_used_mb=2048.0).scope == "device"


def test_evaluation_harness_latency_keys_are_unchanged() -> None:
    """Pins the prefactor: the eval harness's latency stub still emits exactly its four keys."""
    from mlops_cv.eval.evaluate import benchmark_latency

    class _StubModel:
        def predict(self, src: str, verbose: bool = True) -> None:
            return None

    out = benchmark_latency(_StubModel(), ["a.jpg", "b.jpg"], warmup=1, runs=3)  # type: ignore[arg-type]
    assert set(out) == {
        "latency/p50_ms",
        "latency/p95_ms",
        "latency/mean_ms",
        "latency/n",
    }
    assert out["latency/n"] == 3.0


def test_evaluation_harness_latency_without_images_is_zeros() -> None:
    from mlops_cv.eval.evaluate import benchmark_latency

    out = benchmark_latency(object(), [], runs=5)  # type: ignore[arg-type]
    assert out == {
        "latency/p50_ms": 0.0,
        "latency/p95_ms": 0.0,
        "latency/mean_ms": 0.0,
        "latency/n": 0.0,
    }


def test_memory_delta_reports_what_the_variant_added() -> None:
    before = GpuMemory(used_mb=268.0, scope="process")
    after = GpuMemory(used_mb=354.0, scope="process")
    assert memory_delta(before, after) == pytest.approx(86.0)


def test_memory_delta_refuses_to_mix_scopes() -> None:
    """A process reading minus a device-wide one subtracts unrelated quantities."""
    assert memory_delta(GpuMemory(1416.0, "device"), GpuMemory(370.0, "process")) is None


def test_memory_delta_without_a_reading_is_none() -> None:
    assert memory_delta(None, GpuMemory(370.0, "process")) is None
    assert memory_delta(GpuMemory(268.0, "process"), None) is None
