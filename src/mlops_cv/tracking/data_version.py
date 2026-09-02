"""The data version: a Profile published to tracking as a run of its own."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlops_cv.config import Settings, get_settings
from mlops_cv.pipelines import profiling
from mlops_cv.tracking import client
from mlops_cv.tracking.metric_keys import data_metrics
from mlops_cv.tracking.records import tag_filter

if TYPE_CHECKING:
    from types import ModuleType

logger = logging.getLogger(__name__)

# Where the Profile directory lands on its data version's run.
PROFILE_ARTIFACT_PATH = "profile"

# Tags on a data-version run: its identity, and what it describes.
STAMP_TAG = "data.stamp"
SUBSET_TAG = "data.subset"
MANIFEST_TAG = "data.manifest_sha256"

# The tag a TRAINING run carries: the id of the data version it learned from.
RUN_TAG = "data.version"


def stamp_id(stamp: str) -> str:
    """The identity of one data version: a content hash of its canonical provenance stamp."""
    return hashlib.sha256(stamp.encode("utf-8")).hexdigest()


def profile_uri(run_id: str) -> str:
    """The published Profile directory's artifact URI on a data version's run."""
    return f"runs:/{run_id}/{PROFILE_ARTIFACT_PATH}"


def run_name(subset_name: str, identity: str) -> str:
    """A data version's run name — readable in the UI, unique per identity."""
    return f"data-{subset_name}-{identity[:8]}"


def find_data_version(registry: Any, experiment_id: str, identity: str) -> str | None:
    """The run id of the data version with this identity, or ``None`` if there is none."""
    found = registry.search_runs([experiment_id], filter_string=tag_filter(STAMP_TAG, identity))
    return next((run.info.run_id for run in found), None)


def _session(settings: Settings | None, mlflow: ModuleType | None) -> tuple[Settings, ModuleType]:
    """The resolved settings and a connected ``mlflow`` module (injected ones win)."""
    resolved = settings if settings is not None else get_settings()
    return resolved, mlflow if mlflow is not None else client.connect(resolved)


def _experiment_id(mlflow: ModuleType, settings: Settings) -> str:
    return mlflow.get_experiment_by_name(settings.mlflow.experiment).experiment_id


def publish_profile(
    profile_dir: str | Path,
    stamp: str,
    settings: Settings | None = None,
    *,
    mlflow: ModuleType | None = None,
) -> str:
    """Publish a Profile directory as its data version's run; return that run's id."""
    resolved, mlflow = _session(settings, mlflow)
    directory = Path(profile_dir)
    profile = json.loads((directory / profiling.PROFILE_JSON).read_text(encoding="utf-8"))
    identity = stamp_id(stamp)
    subset_name = directory.parent.name

    existing = find_data_version(mlflow.MlflowClient(), _experiment_id(mlflow, resolved), identity)
    started = (
        mlflow.start_run(run_id=existing)
        if existing
        else mlflow.start_run(run_name=run_name(subset_name, identity))
    )
    with started as run:
        mlflow.set_tags(
            {
                STAMP_TAG: identity,
                SUBSET_TAG: subset_name,
                MANIFEST_TAG: profile["source_manifest_sha256"],
            }
        )
        mlflow.log_params(profile["params"])
        mlflow.log_metrics(data_metrics(profiling.profile_counts(profile)))
        mlflow.log_artifacts(str(directory), artifact_path=PROFILE_ARTIFACT_PATH)
        run_id = run.info.run_id
    logger.info(
        f"data version {run_id} ({'reused' if existing else 'created'}) "
        f"carries the profile of {subset_name}"
    )
    return run_id


def published_run(
    subset_dir: str | Path,
    settings: Settings | None = None,
    *,
    mlflow: ModuleType | None = None,
) -> str | None:
    """The data version describing ``subset_dir``'s current data, or ``None`` if unpublished."""
    resolved, mlflow = _session(settings, mlflow)
    try:
        identity = stamp_id(profiling.profile_identity(subset_dir))
    except FileNotFoundError:
        return None
    return find_data_version(mlflow.MlflowClient(), _experiment_id(mlflow, resolved), identity)
