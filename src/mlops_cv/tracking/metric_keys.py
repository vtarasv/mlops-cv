"""The run-metric key vocabulary: the single author of every metric-key string.

Metric keys are a persistence contract, not just code vocabulary: the promotion gate reads a
champion's *historical* run metrics back by these exact strings, so a renamed key doesn't crash —
it strands every previously recorded model and silently degrades the champion/challenger
comparison to a bootstrap win. Every producer (evaluation, the per-epoch training callback) and
every reader (gate floors, report tables) builds keys through this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Headline detection metrics, in display order, keyed by the ultralytics ``results.box`` attribute.
HEADLINE_ATTRS: dict[str, str] = {
    "precision": "mp",
    "recall": "mr",
    "mAP50": "map50",
    "mAP50-95": "map",
}

# The primary metric: the one headline reading the champion/challenger comparison branches on.
PRIMARY = "mAP50-95"

# The prefix ultralytics' MLflow callback logs per-epoch val metrics under.
TRAIN_PREFIX = "metrics"

OPTIMIZE_PREFIX = "optimize"

# The namespace a data version records its Profile's headline counts under.
DATA_PREFIX = "data"

# The eval harness's single-configuration latency namespace (``latency/p50_ms`` ...).
LATENCY_PREFIX = "latency"


def metric_key(prefix: str, name: str) -> str:
    """The tracking-run key for one reading: ``{prefix}/{name}``."""
    return f"{prefix}/{name}"


def variant_prefix(variant: str) -> str:
    """The reserved namespace for one Serving variant's readings: ``optimize/<variant>``."""
    return metric_key(OPTIMIZE_PREFIX, variant)


def variant_latency_prefix(variant: str) -> str:
    """Where a variant's latency readings live: ``optimize/<variant>/latency``."""
    return metric_key(variant_prefix(variant), "latency")


def variant_device_latency_prefix(variant: str, device_label: str) -> str:
    """Where an on-device harness's latency readings live:
    ``optimize/<variant>/latency/<label>``."""
    return metric_key(variant_latency_prefix(variant), device_label)


def variant_speed_prefix(variant: str) -> str:
    """Where a variant's per-stage breakdown lives: ``optimize/<variant>/speed``."""
    return metric_key(variant_prefix(variant), "speed")


def data_metrics(counts: Mapping[str, float]) -> dict[str, float]:
    """Map a data version's headline counts to their run-metric keys: ``data/<name>``."""
    return {metric_key(DATA_PREFIX, name): float(value) for name, value in counts.items()}


def headline_metrics(box: Any, prefix: str = "test") -> dict[str, float]:
    """Map an ultralytics ``results.box`` to ``{prefix/precision, recall, mAP50, mAP50-95}``.

    ``box`` is duck-typed (anything exposing ``mp``/``mr``/``map50``/``map``), so callers/tests can
    pass a stub and CI never imports ultralytics.
    """
    return {
        metric_key(prefix, name): float(getattr(box, attr)) for name, attr in HEADLINE_ATTRS.items()
    }


def per_class_metrics(
    maps: Sequence[float] | NDArray,
    ap_class_index: Sequence[int] | NDArray,
    names: Mapping[int, str],
    *,
    prefix: str = metric_key(TRAIN_PREFIX, PRIMARY),
) -> dict[str, float]:
    """Map per-class mAP50-95 to ``{prefix}/<class-name>`` for the classes actually evaluated.

    Iterates ``ap_class_index`` (not ``maps``) on purpose: ``maps`` is class-id-indexed and
    ultralytics back-fills absent classes with the overall mAP.
    """
    return {metric_key(prefix, names[int(c)]): float(maps[int(c)]) for c in ap_class_index}
