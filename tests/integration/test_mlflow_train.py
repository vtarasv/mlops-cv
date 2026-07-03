"""Integration tests for train.py's MLflow glue (``_register_model``, ``_log_dataset``).

Registration produces the file-source versions that ``_resolve_model`` consumes, so this exercises
the full register → resolve round-trip — including that the weight bytes survive the object-store
round-trip byte-for-byte — plus dataset-lineage logging. Requires the MLflow stack up and the
``gpu`` uv group.
"""

from __future__ import annotations

import contextlib
import uuid

import pytest

from mlops_cv.eval.evaluate import _resolve_model
from mlops_cv.training.train import _log_dataset, _register_model

mlflow = pytest.importorskip("mlflow")  # absent under CI's --no-group gpu -> skip the module

pytestmark = pytest.mark.docker

STUB_WEIGHTS = b"stub-weights"


def test_register_model_registers_file_source_version_and_round_trips(registry, tmp_path) -> None:
    name = f"register-test-{uuid.uuid4().hex[:8]}"
    weights = tmp_path / "best.pt"
    weights.write_bytes(STUB_WEIGHTS)
    try:
        with mlflow.start_run() as run:
            mlflow.log_artifact(str(weights), artifact_path="weights")
            _register_model(run, name)
            returned = _register_model(run, name)  # 2nd model-create is suppressed, not raised
        versions = registry.search_model_versions(f"name='{name}'")
        assert len(versions) == 2  # one version per call; the duplicate model-create didn't crash
        latest = max(versions, key=lambda v: int(v.version))
        assert returned == latest.version  # the returned version feeds the orchestrator handoff
        assert latest.source.endswith("weights/best.pt")  # a bare-file source (our pattern)
        assert latest.run_id == run.info.run_id
        # register -> resolve round-trip: the version we just made downloads back to a .pt
        path, _ = _resolve_model(f"models:/{name}/{latest.version}")
        assert path.is_file() and path.suffix == ".pt"
        assert path.read_bytes() == STUB_WEIGHTS  # bytes survive the object-store round-trip
    finally:
        with contextlib.suppress(Exception):
            registry.delete_registered_model(name)


def test_log_dataset_logs_input_and_sha_tag(registry, tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("name,split\na.jpg,train\nb.jpg,val\n", encoding="utf-8")
    with mlflow.start_run() as run:
        _log_dataset(mlflow, manifest)
        run_id = run.info.run_id
    fetched = registry.get_run(run_id)
    assert fetched.data.tags.get("dataset_sha")  # content-hash tag for lineage
    assert fetched.inputs.dataset_inputs  # the manifest was logged as a tracked dataset input
