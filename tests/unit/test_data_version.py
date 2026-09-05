"""The data version: publishing a Profile as a tracked run, and finding it again."""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlops_cv.config import Settings
from mlops_cv.data.subset import MANIFEST_FILENAME
from mlops_cv.pipelines import profiling, provenance
from mlops_cv.tracking import data_version
from mlops_cv.tracking.data_version import (
    PROFILE_ARTIFACT_PATH,
    STAMP_TAG,
    SUBSET_TAG,
    profile_uri,
    publish_profile,
    published_run,
    stamp_id,
)


class FakeRun:
    def __init__(self, run_id: str, experiment_id: str) -> None:
        self.info = SimpleNamespace(run_id=run_id, experiment_id=experiment_id)
        self.data = SimpleNamespace(tags={}, params={}, metrics={})


class FakeMlflow:
    """A stand-in for the connected ``mlflow`` module, recording every call."""

    EXPERIMENT_ID = "7"

    def __init__(self) -> None:
        self.runs: dict[str, FakeRun] = {}
        self.active: FakeRun | None = None
        self.started: list[dict[str, str | None]] = []
        self.artifacts: list[tuple[str, str]] = []
        self._counter = 0
        outer = self

        class _Client:
            def search_runs(self, experiment_ids: list[str], filter_string: str = "") -> list:
                return outer.search(experiment_ids, filter_string)

        self.MlflowClient = _Client

    # -- the module surface the publisher uses -------------------------------------------
    def get_experiment_by_name(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(experiment_id=self.EXPERIMENT_ID, name=name)

    @contextlib.contextmanager
    def start_run(self, run_id: str | None = None, run_name: str | None = None):  # noqa: ANN201
        self.started.append({"run_id": run_id, "run_name": run_name})
        if run_id is None:
            self._counter += 1
            run_id = f"data-run-{self._counter}"
            self.runs[run_id] = FakeRun(run_id, self.EXPERIMENT_ID)
        self.active = self.runs[run_id]
        try:
            yield self.active
        finally:
            self.active = None

    def set_tags(self, tags: dict[str, str]) -> None:
        assert self.active is not None
        self.active.data.tags.update(tags)

    def log_params(self, params: dict) -> None:
        assert self.active is not None
        self.active.data.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        assert self.active is not None
        self.active.data.metrics.update(metrics)

    def log_artifacts(self, local_dir: str, artifact_path: str) -> None:
        assert self.active is not None
        self.artifacts.append((local_dir, artifact_path))

    # -- the registry surface --------------------------------------------------------------
    def search(self, experiment_ids: list[str], filter_string: str) -> list[FakeRun]:
        match = re.fullmatch(r"tags\.(\S+) = '(.*)'", filter_string)
        assert match, f"unsupported filter {filter_string!r}"
        key, value = match.groups()
        return [
            run
            for run in self.runs.values()
            if run.info.experiment_id in experiment_ids and run.data.tags.get(key) == value
        ]


PROFILE = {
    "schema_version": 2,
    "source_manifest_sha256": "abc",
    "params": profiling.PROFILE_PARAMS,
    "dataset": {"n_frames": 1238, "n_boxes": 4000, "n_duplicate_clusters": 2, "n_flagged": 9},
    "splits": {},
    "classes": {},
    "baseline": {"split": "train", "n_sequences": 56},
}


@pytest.fixture
def subset(tmp_path: Path) -> Path:
    """A subset dir with a manifest and a profile directory, as the pipeline leaves it."""
    root = tmp_path / "visdrone-vid-small"
    profile_dir = root / profiling.PROFILE_DIRNAME
    profile_dir.mkdir(parents=True)
    (root / MANIFEST_FILENAME).write_text("split,sequence\ntrain,seqA\n", encoding="utf-8")
    (profile_dir / profiling.PROFILE_JSON).write_text(json.dumps(PROFILE), encoding="utf-8")
    (profile_dir / profiling.DRIFT_BASELINE_CSV).write_text(
        f"{','.join(profiling.BASELINE_FIELDS)}\nseqA,3,10.0,1.0,50.0\n", encoding="utf-8"
    )
    return root


def _publish(subset: Path, mlflow: FakeMlflow) -> str:
    return publish_profile(
        subset / profiling.PROFILE_DIRNAME,
        profiling.profile_identity(subset),
        Settings(),
        mlflow=mlflow,  # type: ignore
    )


def test_publishes_the_whole_profile_directory(subset: Path) -> None:
    """The baseline is a sibling CSV, so the *directory* is the artifact — not the JSON."""
    mlflow = FakeMlflow()
    run_id = _publish(subset, mlflow)

    assert mlflow.artifacts == [(str(subset / profiling.PROFILE_DIRNAME), PROFILE_ARTIFACT_PATH)]
    run = mlflow.runs[run_id]
    assert run.data.params == profiling.PROFILE_PARAMS
    assert run.data.metrics == {
        "data/n_frames": 1238.0,
        "data/n_boxes": 4000.0,
        "data/n_duplicate_clusters": 2.0,
        "data/n_flagged": 9.0,
        "data/n_baseline_sequences": 56.0,
    }
    assert run.data.tags[SUBSET_TAG] == "visdrone-vid-small"
    assert run.data.tags[STAMP_TAG] == stamp_id(profiling.profile_identity(subset))


def test_unchanged_data_reuses_its_data_version(subset: Path) -> None:
    """Every orchestrated pass may re-profile; identical data must not litter the experiment."""
    mlflow = FakeMlflow()
    first = _publish(subset, mlflow)
    second = _publish(subset, mlflow)

    assert second == first
    assert len(mlflow.runs) == 1
    assert mlflow.started[-1] == {"run_id": first, "run_name": None}  # reopened, not created


def test_changed_manifest_is_a_new_data_version(subset: Path) -> None:
    mlflow = FakeMlflow()
    first = _publish(subset, mlflow)
    (subset / MANIFEST_FILENAME).write_text("split,sequence\ntrain,seqB\n", encoding="utf-8")
    second = _publish(subset, mlflow)

    assert second != first
    assert len(mlflow.runs) == 2


def test_changed_profiling_parameter_is_a_new_data_version(subset: Path) -> None:
    """The stamp is the identity, so a parameter change splits the data version like data does."""
    mlflow = FakeMlflow()
    first = _publish(subset, mlflow)
    changed = provenance.stamp_payload(
        provenance.manifest_sha256(subset / MANIFEST_FILENAME),
        {**profiling.PROFILE_PARAMS, "blur_var": 999.0},
    )
    second = publish_profile(subset / profiling.PROFILE_DIRNAME, changed, Settings(), mlflow=mlflow)  # type: ignore

    assert second != first
    assert len(mlflow.runs) == 2


def test_stamp_identity_is_the_canonical_payload_hash() -> None:
    payload = provenance.stamp_payload("abc", {"a": 1})
    assert stamp_id(payload) == stamp_id(provenance.stamp_payload("abc", {"a": 1}))
    assert stamp_id(payload) != stamp_id(provenance.stamp_payload("abd", {"a": 1}))
    assert len(stamp_id(payload)) == 64


def test_published_run_finds_the_version_describing_the_subset_on_disk(subset: Path) -> None:
    """What training links: the data version of the data it is about to train on."""
    mlflow = FakeMlflow()
    run_id = _publish(subset, mlflow)
    assert published_run(subset, Settings(), mlflow=mlflow) == run_id  # type: ignore


def test_published_run_is_none_when_the_data_moved_on(subset: Path) -> None:
    """A Subset whose Profile was never published — or is stale — links nothing, quietly."""
    mlflow = FakeMlflow()
    assert published_run(subset, Settings(), mlflow=mlflow) is None  # type: ignore

    _publish(subset, mlflow)
    (subset / MANIFEST_FILENAME).write_text("split,sequence\ntrain,seqB\n", encoding="utf-8")
    assert published_run(subset, Settings(), mlflow=mlflow) is None  # type: ignore


def test_published_run_survives_a_subset_with_no_manifest(tmp_path: Path) -> None:
    """Model production must not fail because a Subset was never profiled."""
    assert published_run(tmp_path, Settings(), mlflow=FakeMlflow()) is None  # type: ignore


def test_profile_uri_addresses_the_published_directory() -> None:
    assert profile_uri("abc123") == f"runs:/abc123/{PROFILE_ARTIFACT_PATH}"


def test_the_training_link_tag_is_spelled_once() -> None:
    """A cross-process contract: training writes it, the monitor reads it."""
    assert data_version.RUN_TAG == "data.version"
    assert STAMP_TAG == "data.stamp"
    assert PROFILE_ARTIFACT_PATH == "profile"
