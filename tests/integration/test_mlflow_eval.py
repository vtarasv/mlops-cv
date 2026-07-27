"""Integration tests for the eval MLflow glue against the live stack: ``resolve_model`` (download)
and ``registered_version`` (which registered version ``--promote`` aliases).

Requires the MLflow stack up and the ``gpu`` uv group.
"""

from __future__ import annotations

import contextlib
import uuid

import pytest

from mlops_cv.tracking.resolve import registered_version, resolve_model

mlflow = pytest.importorskip("mlflow")  # absent under CI's --no-group gpu -> skip the module

pytestmark = pytest.mark.docker


@pytest.fixture(scope="module")
def file_source_model(registry, tmp_path_factory):
    """Register a throwaway version whose source is a bare ``weights/best.pt`` file.

    Yields ``(name, version, run_id)`` and deletes the registered model afterward.
    """
    name = f"resolve-test-{uuid.uuid4().hex[:8]}"
    weights = tmp_path_factory.mktemp("w") / "best.pt"
    weights.write_bytes(b"not-a-real-model")
    with mlflow.start_run() as run:
        mlflow.log_artifact(str(weights), artifact_path="weights")
    registry.create_registered_model(name)
    version = registry.create_model_version(
        name=name, source=f"{run.info.artifact_uri}/weights/best.pt", run_id=run.info.run_id
    )
    registry.set_registered_model_alias(name, "champion", version.version)
    yield name, version.version, run.info.run_id
    with contextlib.suppress(Exception):
        registry.delete_registered_model(name)


def _assert_downloads_pt(ref: str) -> None:
    path, slug = resolve_model(ref)
    assert path.is_file() and path.suffix == ".pt", f"{ref} -> {path}"
    assert slug


def test_resolve_by_alias(file_source_model) -> None:
    name, _, _ = file_source_model
    _assert_downloads_pt(f"models:/{name}@champion")  # the champion/challenger happy path


def test_resolve_by_version(file_source_model) -> None:
    name, version, _ = file_source_model
    _assert_downloads_pt(f"models:/{name}/{version}")


def test_resolve_by_run_uri(file_source_model) -> None:
    _, _, run_id = file_source_model
    _assert_downloads_pt(f"runs:/{run_id}/weights/best.pt")


def test_promotion_version_by_alias(file_source_model) -> None:
    name, version, _ = file_source_model
    assert registered_version(f"models:/{name}@champion", name) == version


def test_promotion_version_by_run_uri(file_source_model) -> None:
    # The run-id search filter — the exact syntax a mock can't validate.
    name, version, run_id = file_source_model
    assert registered_version(f"runs:/{run_id}/weights/best.pt", name) == version
