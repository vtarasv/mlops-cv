"""The champion, resolved: alias -> model version -> the training run that produced it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlops_cv.config import Settings
from mlops_cv.startup import StartupError
from mlops_cv.tracking import client

if TYPE_CHECKING:
    from mlflow.entities import Run
    from mlflow.entities.model_registry import ModelVersion

# What fixes every "nothing has been promoted" state.
TRAIN_HINT = "Train and promote a model first (`make train`, then evaluation promotes a winner)."


def resolve(
    settings: Settings, version: str | None = None, *, registry: Any | None = None
) -> ModelVersion:
    """The champion — or, given a number, that registered version — as the full entity.

    The entity, not its number: its tags carry the published-artifact addresses and its
    ``run_id`` names the training run. Raises :class:`StartupError` when the alias points at
    nothing (the ordinary state of a fresh install) or the pinned version does not exist.
    """
    from mlflow.exceptions import MlflowException

    name = settings.mlflow.registered_model
    alias = settings.mlflow.champion_alias
    reg = client.registry(settings, injected=registry)
    try:
        if version:
            return reg.get_model_version(name, version)
        return reg.get_model_version_by_alias(name, alias)
    except MlflowException as exc:
        hint = (
            f"'{name}' has no version {version} — check the number against the registry."
            if version
            else f"no '{alias}' alias on '{name}' — {TRAIN_HINT}"
        )
        raise StartupError(f"{hint} ({exc})") from exc


def training_run(version: ModelVersion, *, registry: Any | None = None) -> Run:
    """The run that produced ``version`` — where its metrics, tags and record runs live.

    Raises :class:`StartupError` when the version names no run (registered outside a tracked
    run) or the run it names is gone (pruned after promotion).
    """
    from mlflow.exceptions import MlflowException

    if not version.run_id:
        raise StartupError(
            f"model version {version.version} names no training run, so nothing records how it "
            f"was made. {TRAIN_HINT}"
        )
    try:
        return client.registry(injected=registry).get_run(version.run_id)
    except MlflowException as exc:
        raise StartupError(
            f"model version {version.version}'s training run {version.run_id} cannot be read "
            f"({exc}) — the run behind the promoted model is gone. {TRAIN_HINT}"
        ) from exc
