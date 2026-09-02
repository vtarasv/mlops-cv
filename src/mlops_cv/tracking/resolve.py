"""Resolve model URIs against MLflow: artifact download + registered-version lookup.

Shared by every consumer of a registered model (evaluation, optimization, serving, monitoring):
the same ``models:/`` / ``runs:/`` grammar resolves everywhere, and every published artifact —
weights, graph, engine, NCNN directory, Profile — comes down through :func:`download`. mlflow
imports stay lazy so the module is importable under CI's CPU-only sync.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlops_cv.tracking import client

if TYPE_CHECKING:
    from mlflow.entities.model_registry import ModelVersion


def slug(text: str) -> str:
    """A filesystem/run-name-safe slug for a model reference."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def download(uri: str) -> Path:
    """Pull a published artifact — a file or a directory — to local disk.

    Proxied through the tracking server, so the only thing a client needs is the tracking URI.
    """
    from mlflow.artifacts import download_artifacts

    return Path(download_artifacts(artifact_uri=uri))


def resolve_model(model_ref: str) -> tuple[Path, str]:
    """Download a model's ``best.pt`` from MLflow → ``(local path, run-name slug)``.

    ``model_ref`` is an MLflow URI: a run URI ``runs:/<id>/<path>`` or a registry ref
    ``models:/<name>@<alias>`` / ``models:/<name>/<version>``. A registry ref is redirected to its
    version's ``source`` artifact URI and downloaded from there.
    """
    if model_ref.startswith("models:/"):
        spec = model_ref.removeprefix("models:/")
        registry = client.registry()
        version: ModelVersion = (
            registry.get_model_version_by_alias(*spec.split("@", 1))
            if "@" in spec
            else registry.get_model_version(*spec.rsplit("/", 1))
        )
        model_ref, ref_slug = version.source, slug(spec)  # type: ignore[union-attr]
    else:
        ref_slug = slug(model_ref.split(":/", 1)[1])
    return download(model_ref), ref_slug


def run_id_from_uri(uri: str) -> str | None:
    """The run id a ``runs:/`` URI addresses; ``None`` for any other reference."""
    if not uri.startswith("runs:/"):
        return None
    return uri.removeprefix("runs:/").split("/", 1)[0]


def model_version(
    model_ref: str, registered_model: str, *, registry: Any | None = None
) -> ModelVersion | None:
    """The registered ``ModelVersion`` a model URI refers to; ``None`` when it maps to none.

    The full entity, not just its number: a version's tags carry the published-artifact
    addresses, its ``run_id`` names the training run, and refetching those one field at a time
    is how callers grow their own registry clients. ``models:/`` refs resolve directly (a
    missing alias/version raises, as it would on use); ``runs:/`` refs are searched by run id;
    anything else (a bare local path) is ``None``.
    """
    if model_ref.startswith("models:/"):
        spec = model_ref.removeprefix("models:/")
        if "@" in spec:
            return client.registry(injected=registry).get_model_version_by_alias(
                *spec.split("@", 1)
            )
        if "/" in spec:
            return client.registry(injected=registry).get_model_version(*spec.rsplit("/", 1))
        return None
    if run_id := run_id_from_uri(model_ref):
        found = client.registry(injected=registry).search_model_versions(
            f"run_id='{run_id}' and name='{registered_model}'"
        )
        return found[0] if found else None
    return None


def registered_version(model_ref: str, registered_model: str) -> str | None:
    """The registered version *number* a model URI refers to; ``None`` for a bare local path.

    A ``models:/<name>/<version>`` ref already carries the number, so it stays a pure string
    parse (no registry query); everything else goes through :func:`model_version`.
    """
    if model_ref.startswith("models:/"):
        spec = model_ref.removeprefix("models:/")
        if "@" not in spec and "/" in spec:
            return spec.rsplit("/", 1)[1]
    version = model_version(model_ref, registered_model)
    return version.version if version is not None else None
