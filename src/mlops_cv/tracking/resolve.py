"""Resolve model URIs against MLflow: weights download + registered-version lookup.

Shared by every consumer of a registered model (evaluation, streaming inference): the same
``models:/`` / ``runs:/`` grammar resolves everywhere. mlflow imports stay lazy so the module
is importable under CI's CPU-only sync.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow.entities.model_registry import ModelVersion


def slug(text: str) -> str:
    """A filesystem/run-name-safe slug for a model reference."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def resolve_model(model_ref: str) -> tuple[Path, str]:
    """Download a model's ``best.pt`` from MLflow → ``(local path, run-name slug)``.

    ``model_ref`` is an MLflow URI: a run URI ``runs:/<id>/<path>`` or a registry ref
    ``models:/<name>@<alias>`` / ``models:/<name>/<version>``. A registry ref is redirected to its
    version's ``source`` artifact URI and downloaded from there.
    """
    from mlflow.artifacts import download_artifacts

    if model_ref.startswith("models:/"):
        from mlflow import MlflowClient

        spec = model_ref.removeprefix("models:/")
        registry = MlflowClient()
        version: ModelVersion = (
            registry.get_model_version_by_alias(*spec.split("@", 1))
            if "@" in spec
            else registry.get_model_version(*spec.rsplit("/", 1))
        )
        model_ref, ref_slug = version.source, slug(spec)  # type: ignore[union-attr]
    else:
        ref_slug = slug(model_ref.split(":/", 1)[1])
    return Path(download_artifacts(artifact_uri=model_ref)), ref_slug


def registered_version(model_ref: str, registered_model: str) -> str | None:
    """The registered version a model URI refers to: parsed from a ``models:/`` URI or found by
    run id for a ``runs:/`` URI. A bare local path has no registered version -> ``None``."""
    from mlflow import MlflowClient

    registry = MlflowClient()
    if model_ref.startswith("models:/"):
        spec = model_ref.removeprefix("models:/")
        if "@" in spec:
            name, alias = spec.split("@", 1)
            return registry.get_model_version_by_alias(name, alias).version
        if "/" in spec:
            return spec.rsplit("/", 1)[1]
    if model_ref.startswith("runs:/"):
        run_id = model_ref.removeprefix("runs:/").split("/", 1)[0]
        found = registry.search_model_versions(f"run_id='{run_id}' and name='{registered_model}'")
        if found:
            return found[0].version
    return None
