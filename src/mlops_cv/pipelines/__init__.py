"""Beam data pipelines (ingest, profile) and the pure profiling/provenance modules they share."""

from __future__ import annotations

import shutil
import sys

# Prism uses this as a gRPC deadline on the whole job, not just submission (300 s default).
JOB_SERVER_TIMEOUT_S = 24 * 3600


def query_counters(result, namespace: str) -> dict[str, int]:  # noqa: ANN001 - Beam PipelineResult
    """A finished pipeline's counters under ``namespace``, by name."""
    from apache_beam.metrics.metric import MetricsFilter

    metrics = result.metrics().query(MetricsFilter().with_namespace(namespace))
    return {m.key.metric.name: m.result for m in metrics["counters"]}


def beam_argv(argv: list[str] | None) -> list[str]:
    """Pipeline argv with the raised job deadline first, so an explicit flag still wins."""
    return [
        f"--job_server_timeout={JOB_SERVER_TIMEOUT_S}",
        *(sys.argv[1:] if argv is None else argv),
    ]


def stream_copy(src: str, dest: str) -> None:
    """Copy one file across filesystems (``FileSystems.copy`` dispatches on the source's)."""
    from apache_beam.io.filesystems import FileSystems

    with FileSystems.open(src) as fin, FileSystems.create(dest) as fout:
        shutil.copyfileobj(fin, fout)


def delete_tree(path: str) -> None:
    """Delete a directory tree, local or on an object store; a missing one is fine."""
    from apache_beam.io.filesystems import FileSystems

    if not FileSystems.get_filesystem(path).has_dirs():
        # A blob prefix is deleted only with the trailing '/' (without it: a silent no-op);
        # on a local path that '/' would glob files only, and fail on a dir of dirs.
        FileSystems.delete([path.rstrip("/") + "/"])
    elif FileSystems.exists(path):
        FileSystems.delete([path])
