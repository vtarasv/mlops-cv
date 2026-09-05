"""Record runs: a child of a training run that one producer appends to, found or created."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

# MLflow nesting is this tag; ``mlflow.start_run(nested=True)`` sets the same one.
PARENT_TAG = "mlflow.parentRunId"


def tag_filter(key: str, value: str) -> str:
    """The one spelling of a tag-equality search the tracking server parses."""
    return f"tags.{key} = '{value}'"


def open_record(
    registry: Any,
    parent_run_id: str,
    marker: str,
    *,
    run_name: str,
    tags: Mapping[str, str],
) -> str:
    """The record's run id: found by ``marker`` under the parent, else created terminated.

    ``tags`` must carry ``marker`` — it is the record's identity, and a record created without
    it could never be found again.
    """
    if marker not in tags:
        raise ValueError(f"record tags must carry the marker {marker!r}; got {sorted(tags)}")
    experiment_id = registry.get_run(parent_run_id).info.experiment_id  # nested = same experiment
    children = registry.search_runs(
        [experiment_id], filter_string=tag_filter(PARENT_TAG, parent_run_id)
    )
    if found := next((r.info.run_id for r in children if marker in r.data.tags), None):
        return found
    run = registry.create_run(
        experiment_id, tags={PARENT_TAG: parent_run_id, **tags}, run_name=run_name
    )
    run_id = run.info.run_id
    registry.set_terminated(run_id, "FINISHED")
    logger.info(f"opened the {run_name} record {run_id} under run {parent_run_id}")
    return run_id
