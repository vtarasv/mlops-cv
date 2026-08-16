"""What the monitor judges against: the champion's own drift baseline, resolved at startup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mlops_cv.config import Settings
from mlops_cv.pipelines import profiling
from mlops_cv.serving.errors import StartupError
from mlops_cv.serving.resolve import champion_version
from mlops_cv.streaming.messages import ModelInfo
from mlops_cv.tracking.data_version import RUN_TAG, profile_uri

if TYPE_CHECKING:
    from mlflow.entities.model_registry import ModelVersion

    from mlops_cv.pipelines.profiling import BaselineScene

REPROFILE_HINT = "re-profile the training data (`make profile`), then retrain (`make train`)"


@dataclass(frozen=True)
class ResolvedBaseline:
    """A champion's drift baseline plus the identities behind it."""

    scenes: list[BaselineScene]
    model: ModelInfo
    data_run_id: str


def training_run_tags(run_id: str) -> dict[str, str]:
    """The tags of a training run — where its data-version link lives."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    try:
        return dict(MlflowClient().get_run(run_id).data.tags)
    except MlflowException as exc:
        raise StartupError(
            f"the champion's training run {run_id} cannot be read ({exc}) — the run behind the "
            "promoted model is gone. Train and promote a model (`make train`)."
        ) from exc


def download_profile(uri: str) -> Path:
    """Pull a published Profile directory to local disk (proxied through the tracking server)."""
    from mlflow.artifacts import download_artifacts

    return Path(download_artifacts(artifact_uri=uri))


def data_version_id(version: ModelVersion) -> str:
    """The data version a model version's training run learned from."""
    if not version.run_id:
        # A version registered outside a tracked run: the walk has no second step to take.
        raise StartupError(
            f"the champion (model version {version.version}) names no training run, so nothing "
            "records which data it learned from. Train and promote a model (`make train`)."
        )
    run_id = training_run_tags(version.run_id).get(RUN_TAG)
    if not run_id:
        raise StartupError(
            f"the champion's training run {version.run_id} carries no '{RUN_TAG}' tag — model "
            f"version {version.version} was trained without a published profile of its data, so "
            f"there is nothing to call normal. Fix: {REPROFILE_HINT}."
        )
    return run_id


def baseline_scenes(data_run_id: str) -> list[BaselineScene]:
    """The training-scene cloud published by one data version."""
    from mlflow.exceptions import MlflowException

    try:
        local = download_profile(profile_uri(data_run_id))
    except MlflowException as exc:
        raise StartupError(
            f"the champion's data version {data_run_id} carries no profile ({exc}) — the run was "
            f"deleted, so the baseline it held is gone. Fix: {REPROFILE_HINT}."
        ) from exc
    try:
        scenes = profiling.read_baseline(local)
    except FileNotFoundError as exc:
        raise StartupError(
            f"the champion's data version {data_run_id} holds no "
            f"{profiling.DRIFT_BASELINE_CSV} — its profile predates the drift baseline. "
            f"Fix: {REPROFILE_HINT}."
        ) from exc
    except ValueError as exc:
        # The baseline's columns are an emit<->parse contract; a profile written under an older
        # spelling is stale data, not a crash.
        raise StartupError(
            f"the champion's data version {data_run_id} holds a drift baseline this build "
            f"cannot read ({exc}). Fix: {REPROFILE_HINT}."
        ) from exc
    if not scenes:
        raise StartupError(
            f"the champion's data version {data_run_id} holds an empty drift baseline — there "
            f"are no training scenes to judge a window against. Fix: {REPROFILE_HINT}."
        )
    return scenes


def resolve_baseline(settings: Settings) -> ResolvedBaseline:
    """Champion alias -> the drift baseline the monitor scores live windows against."""
    version = champion_version(settings)
    data_run_id = data_version_id(version)
    return ResolvedBaseline(
        scenes=baseline_scenes(data_run_id),
        model=ModelInfo(name=settings.mlflow.registered_model, version=version.version),
        data_run_id=data_run_id,
    )
