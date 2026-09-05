"""Beam data pipelines (ingest, profile) and the pure profiling/provenance modules they share."""

from __future__ import annotations


def query_counters(result, namespace: str) -> dict[str, int]:  # noqa: ANN001 - Beam PipelineResult
    """A finished pipeline's counters under ``namespace``, by name."""
    from apache_beam.metrics.metric import MetricsFilter

    metrics = result.metrics().query(MetricsFilter().with_namespace(namespace))
    return {m.key.metric.name: m.result for m in metrics["counters"]}
