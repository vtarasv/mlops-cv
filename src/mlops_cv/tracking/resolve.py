"""Resolve model URIs against MLflow: artifact download + registered-version lookup.

Shared by every consumer of a registered model (evaluation, optimization, serving, monitoring):
the same ``models:/`` / ``runs:/`` grammar resolves everywhere, and every published artifact —
weights, graph, engine, NCNN directory, Profile — comes down through :func:`download`. mlflow
imports stay lazy so the module is importable under CI's CPU-only sync.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlops_cv.optimize.variants import NCNN, parse_artifact_tag
from mlops_cv.startup import StartupError
from mlops_cv.tracking import client

if TYPE_CHECKING:
    from mlflow.entities.model_registry import ModelVersion


def slug(text: str) -> str:
    """A filesystem/run-name-safe slug for a model reference."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def download(uri: str, dst: Path | None = None) -> Path:
    """Pull a published artifact — a file or a directory — to local disk (``dst``, else a temp dir).

    Proxied through the tracking server, so the only thing a client needs is the tracking URI.
    """
    from mlflow.artifacts import download_artifacts

    return Path(download_artifacts(artifact_uri=uri, dst_path=dst and str(dst)))


def resolve_model(model_ref: str) -> tuple[Path, str]:
    """Download a model's ``best.pt`` from MLflow → ``(local path, run-name slug)``.

    ``model_ref`` is an MLflow URI: a run URI ``runs:/<id>/<path>`` or a registry ref
    ``models:/<name>@<alias>`` / ``models:/<name>/<version>``. A registry ref is redirected to its
    version's ``source`` artifact URI and downloaded from there.
    """
    if model_ref.startswith("models:/"):
        spec = model_ref.removeprefix("models:/")
        model_ref, ref_slug = _lookup(spec, None).source, slug(spec)  # type: ignore[union-attr]
    else:
        ref_slug = slug(model_ref.split(":/", 1)[1])
    return download(model_ref), ref_slug


def run_id_from_uri(uri: str) -> str | None:
    """The run id a ``runs:/`` URI addresses; ``None`` for any other reference."""
    if not uri.startswith("runs:/"):
        return None
    return uri.removeprefix("runs:/").split("/", 1)[0]


def ncnn_artifacts(tags: Mapping[str, str], names: Iterable[str] | None) -> dict[str, str]:
    """A version's published NCNN artifacts, ``{variant name: artifact URI}``, all or ``names``."""
    published = {
        variant.name: uri
        for key, uri in tags.items()
        if (variant := parse_artifact_tag(key)) is not None and variant.runtime == NCNN
    }
    if names is None:
        return published
    wanted = set(names)
    if unknown := wanted - set(published):
        raise ValueError(
            f"not published on this version: {', '.join(sorted(unknown))}. "
            f"Published: {', '.join(sorted(published)) or 'none'}"
        )
    return {name: uri for name, uri in published.items() if name in wanted}


def record_run_id(artifact_uris: Iterable[str]) -> str:
    """The run holding the published artifacts, read out of the URIs the version tags carry.

    Not the version's own training run: the desktop logs its record to a *child* of that run, so
    following the artifact URI is what puts the device's readings beside the desktop's.
    """
    run_ids = {run_id for uri in artifact_uris if (run_id := run_id_from_uri(uri))}
    if len(run_ids) != 1:
        raise ValueError(
            f"expected one record run across the published artifacts, found {sorted(run_ids)} — "
            "re-run the optimization so every artifact comes from one ladder"
        )
    return run_ids.pop()


def _lookup(spec: str, registry: Any | None) -> ModelVersion:
    """The registry entity a ``models:/`` spec (``name@alias`` or ``name/version``) names."""
    reg = client.registry(injected=registry)
    if "@" in spec:
        return reg.get_model_version_by_alias(*spec.split("@", 1))
    return reg.get_model_version(*spec.rsplit("/", 1))


def _check_registry_name(spec: str, registered_model: str) -> None:
    """Refuse a ``models:/`` ref naming another registered model.

    Callers use the *number* they get back against ``registered_model`` (promotion sets the
    alias on it, optimization tags its versions), so ``models:/other/7`` would silently act on
    version 7 of the configured model instead of the one named.
    """
    name = spec.split("@", 1)[0] if "@" in spec else spec.rsplit("/", 1)[0]
    if name != registered_model:
        raise StartupError(
            f"models:/{spec} names registered model '{name}', but this process acts on "
            f"'{registered_model}' (MLFLOW__REGISTERED_MODEL) — pass a ref on that model, "
            "or change the setting."
        )


def model_version(
    model_ref: str, registered_model: str, *, registry: Any | None = None
) -> ModelVersion | None:
    """The registered ``ModelVersion`` a model URI refers to; ``None`` when it maps to none.

    The full entity, not just its number: a version's tags carry the published-artifact
    addresses, its ``run_id`` names the training run, and refetching those one field at a time
    is how callers grow their own registry clients. ``models:/`` refs resolve directly (a
    missing alias/version raises, as it would on use; a ref on *another* registered model is a
    :class:`StartupError`); ``runs:/`` refs are searched by run id; anything else (a bare local
    path) is ``None``.
    """
    if model_ref.startswith("models:/"):
        spec = model_ref.removeprefix("models:/")
        if "@" not in spec and "/" not in spec:
            return None
        _check_registry_name(spec, registered_model)
        return _lookup(spec, registry)
    if run_id := run_id_from_uri(model_ref):
        found = client.registry(injected=registry).search_model_versions(
            f"run_id='{run_id}' and name='{registered_model}'"
        )
        return found[0] if found else None
    return None


def registered_version(model_ref: str, registered_model: str) -> str | None:
    """The registered version *number* a model URI refers to; ``None`` for a bare local path."""
    version = model_version(model_ref, registered_model)
    return version.version if version is not None else None
