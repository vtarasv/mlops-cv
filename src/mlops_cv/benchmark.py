"""The one latency/memory measurement implementation: predict timing, statistics, GPU memory."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mlops_cv.tracking.metric_keys import LATENCY_PREFIX

logger = logging.getLogger(__name__)

DEFAULT_WARMUP = 20
DEFAULT_ITERATIONS = 200

DEFAULT_PERCENTILES: tuple[float, ...] = (50.0, 95.0, 99.0)


def percentiles(samples: Sequence[float], ps: Sequence[float] = (50.0, 95.0)) -> dict[float, float]:
    """Linear-interpolation percentiles of ``samples`` (numpy's default method). Empty -> zeros."""
    if not samples:
        return {float(p): 0.0 for p in ps}
    return {float(p): float(v) for p, v in zip(ps, np.percentile(samples, ps), strict=True)}


@dataclass(frozen=True)
class GpuMemory:
    """A GPU memory reading and how honestly it is scoped.

    ``scope`` is ``"process"`` when the reading is this process's own usage, and ``"device"``
    when it is the whole device's (which includes anything else using the GPU — a display
    server, another container).
    """

    used_mb: float
    scope: str


def time_calls(
    call: Callable[[Any], object],
    items: Sequence[Any],
    *,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
) -> list[float]:
    """Call ``call(item)`` over a cycled ``items`` and return the timed durations in ms.

    The first ``warmup`` calls run but are not timed. Returns ``[]`` for empty ``items``.
    """
    if not items:
        return []
    timings: list[float] = []
    for i in range(warmup + iterations):
        start = time.perf_counter()
        call(items[i % len(items)])
        if i >= warmup:
            timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def latency_metrics(
    timings: Sequence[float],
    *,
    prefix: str = LATENCY_PREFIX,
    ps: Sequence[float] = DEFAULT_PERCENTILES,
) -> dict[str, float]:
    """Summarise timings (ms) as ``{prefix}/p<N>_ms`` + ``mean_ms`` + ``n``; empty -> zeros."""
    pct = percentiles(timings, ps)
    out = {f"{prefix}/p{int(p)}_ms": pct[float(p)] for p in ps}
    out[f"{prefix}/mean_ms"] = sum(timings) / len(timings) if timings else 0.0
    out[f"{prefix}/n"] = float(len(timings))
    return out


def benchmark_model(
    model: Any,
    images: Sequence[str],
    *,
    imgsz: int,
    device: str,
    prefix: str = LATENCY_PREFIX,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    ps: Sequence[float] = DEFAULT_PERCENTILES,
) -> dict[str, float]:
    """Time single-image ``model.predict`` calls at one pinned configuration -> latency metrics.

    ``model`` is duck-typed (anything exposing ``predict``). The measurement configuration
    (``imgsz``, ``device``) is part of the interface on purpose: a latency read at the predict
    call's own defaults is not comparable to the accuracy it is reported beside. End-to-end
    timings — batch 1, pre/post-processing included.
    """
    timings = time_calls(
        lambda src: model.predict(src, imgsz=imgsz, device=device, verbose=False),
        list(images),
        warmup=warmup,
        iterations=iterations,
    )
    return latency_metrics(timings, prefix=prefix, ps=ps)


def select_used_mb(
    process_used_mb: Mapping[int, float], pid: int, device_used_mb: float
) -> GpuMemory:
    """Choose the most honest available reading: this process's usage, else the device's."""
    if pid in process_used_mb:
        return GpuMemory(used_mb=float(process_used_mb[pid]), scope="process")
    return GpuMemory(used_mb=float(device_used_mb), scope="device")


def memory_delta(before: GpuMemory | None, after: GpuMemory | None) -> float | None:
    """How much GPU memory a variant *added*, or ``None`` when that can't be answered honestly."""
    if before is None or after is None or before.scope != after.scope:
        return None
    return after.used_mb - before.used_mb


def read_gpu_memory(device_index: int = 0) -> GpuMemory | None:
    """Current GPU memory in use, per-process where possible; ``None`` when unavailable."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            device_used_mb = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1 << 20)  # type: ignore[attr-defined]
            procs = {
                p.pid: (p.usedGpuMemory or 0) / (1 << 20)
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            }
            return select_used_mb(procs, os.getpid(), device_used_mb)
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        logger.warning(f"GPU memory unavailable: {exc}")
        return None
