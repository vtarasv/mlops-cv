"""DagBag import guard: every DAG under ``dags/`` parses cleanly with the expected topology.
Runs in CI (the ``airflow`` uv group is CI-synced).
"""

from __future__ import annotations

import os
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

# Importing airflow materializes AIRFLOW_HOME (default ~/airflow) as a side effect — point it at
# a throwaway BEFORE the importorskip below.
os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp(prefix="airflow-home-"))
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")

airflow = pytest.importorskip("airflow")

REPO = Path(__file__).resolve().parents[2]
DAGS_DIR = REPO / "dags"
CT_DAGS = {"server": "aerial_object_detection_ct_server", "edge": "aerial_object_detection_ct_edge"}
GPU_TASKS = {"train", "evaluate", "optimize"}


@pytest.fixture(scope="module")
def dagbag():
    from airflow.dag_processing.dagbag import DagBag

    # The CT DAG requires these at parse time (no in-DAG fallbacks). Set them only while the DagBag
    # parses and revert on return, so they don't leak into other test modules' os.environ.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HOST_SUBSET_DIR", str(REPO / "data" / "subset"))
        mp.setenv("HOST_WEIGHTS", str(REPO / "data" / "weights" / "yolo26s.pt"))
        mp.setenv("TRAIN_IMAGE", "mlops-cv-train:test")
        mp.setenv("BEAM_IMAGE", "mlops-cv-beam:test")
        mp.setenv("OPTIMIZE_IMAGE", "mlops-cv-optimize:test")
        mp.setenv("MLFLOW__TRACKING_URI", "http://mlflow:5000")
        mp.setenv("HOST_USER", "1234:5678")
        mp.setenv("HOST_USERNAME", "alice")
        return DagBag(dag_folder=str(DAGS_DIR), include_examples=False)


def test_dags_import_cleanly(dagbag) -> None:
    assert dagbag.import_errors == {}
    assert set(dagbag.dag_ids) == set(CT_DAGS.values())


@pytest.fixture(params=sorted(CT_DAGS))
def target(request) -> str:
    return request.param


@pytest.fixture
def ct(dagbag, target):
    return dagbag.dags[CT_DAGS[target]]  # the parsed in-memory DAG


def test_ct_dag_topology(ct) -> None:
    assert {t.task_id for t in ct.tasks} == {
        "check_subset",
        "build_subset",
        "check_profiled",
        "profile",
        "validate_data",
        "train",
        "parse_train_output",
        "evaluate",
        "gate",
        "promote",
        "skip_promotion",
        "optimize",
    }
    down = {t.task_id: set(t.downstream_task_ids) for t in ct.tasks}
    assert down["check_subset"] == {"build_subset", "check_profiled"}  # self-heal branch
    assert down["build_subset"] == {"check_profiled"}
    assert down["check_profiled"] == {"profile", "validate_data"}  # skip-if-current branch
    assert down["profile"] == {"validate_data"}
    assert down["validate_data"] == {"train"}
    assert down["train"] == {"parse_train_output"}
    assert down["parse_train_output"] == {"evaluate", "promote"}  # promote templates the version
    assert down["evaluate"] == {"gate"}
    assert down["gate"] == {"optimize", "skip_promotion"}  # the champion/challenger branch
    assert down["optimize"] == {"promote"}


def test_ct_dag_branch_joins_survive_skipped_paths(ct) -> None:
    rules = {t.task_id: t.trigger_rule for t in ct.tasks}  # str-enum: compares to its value
    assert rules["check_profiled"] == "none_failed_min_one_success"
    assert rules["validate_data"] == "none_failed_min_one_success"


def test_ct_dag_run_policy(ct) -> None:
    assert ct.catchup is False  # a missed week must not queue a backfill of GPU trainings
    assert ct.max_active_runs == 1  # one CT cycle at a time


def test_params_default_to_the_targets_recipe(ct, target) -> None:
    from mlops_cv.training.train import TARGET_MODELS

    defaults = {name: ct.params[name] for name in ("model", "imgsz", "head", "epochs", "batch")}
    assert defaults == {
        "model": TARGET_MODELS[target].removeprefix("yolo26"),
        "imgsz": 1280,
        "head": "one-to-many",
        "epochs": 30,
        "batch": -1,
    }


def test_the_trigger_form_rejects_values_outside_the_enums(ct) -> None:
    from airflow.sdk.exceptions import ParamValidationError

    params = ct.params
    for name, bad in [("model", "x"), ("imgsz", 1024), ("head", "nms"), ("epochs", 0)]:
        with pytest.raises(ParamValidationError):
            params.get_param(name).resolve(bad)


def test_params_reach_the_containers(ct, target) -> None:
    train = ct.get_task("train").command
    assert train[train.index("--target") + 1] == target
    assert "yolo26{{ params.model }}" in train
    assert {"{{ params.imgsz }}", "{{ params.epochs }}", "{{ params.batch }}"} <= set(train)
    for task_id in ("evaluate", "optimize"):
        cmd = ct.get_task(task_id).command
        assert cmd[cmd.index("--target") + 1] == target
        assert cmd[cmd.index("--head") + 1] == "{{ params.head }}"
        assert "--imgsz" not in cmd  # read from the training run


def test_gpu_tasks_share_the_pool_airflow_init_creates(ct) -> None:
    """Two trainings at once spill label assignment onto the CPU (~10x slower)."""
    assert {t.task_id for t in ct.tasks if t.pool == "gpu"} == GPU_TASKS
    compose = (REPO / "docker-compose" / "docker-compose.airflow.yml").read_text(encoding="utf-8")
    assert "airflow pools set gpu 1 " in compose


def test_train_gets_the_long_timeout(ct) -> None:
    timeouts = {t.task_id: t.execution_timeout for t in ct.tasks if t.task_id in GPU_TASKS}
    assert timeouts == {
        "train": timedelta(hours=3),
        "evaluate": timedelta(hours=2),
        "optimize": timedelta(hours=2),
    }


def test_optimize_sits_between_the_gate_and_promotion(ct) -> None:
    """Only a gate winner earns the GPU minutes; only a servable version becomes the champion."""
    assert ct.get_task("optimize").upstream_task_ids == {"gate"}
    assert ct.get_task("promote").upstream_task_ids == {"optimize", "parse_train_output"}
    assert "optimize" not in ct.get_task("skip_promotion").downstream_task_ids


def test_optimize_task_replies_with_nothing(ct) -> None:
    """Data-prep precedent: no consumer reads a reply, so no reply contract is maintained."""
    assert ct.get_task("optimize").do_xcom_push is False


def test_optimize_task_needs_no_host_store(ct) -> None:
    """Everything the optimize container produces is published to MLflow — an extra host mount
    would mean an artifact silently escaping the registry (and root-owned files on the host)."""
    train_task = ct.get_task("train")
    optimize_task = ct.get_task("optimize")
    assert optimize_task.mounts == train_task.mounts


def test_task_containers_run_as_the_host_user(ct) -> None:
    """Root in a task container writes root-owned files into the host's bind-mounted datasets."""
    from airflow.providers.docker.operators.docker import DockerOperator

    containers = [t for t in ct.tasks if isinstance(t, DockerOperator)]
    assert {t.task_id for t in containers} == {
        "build_subset",
        "profile",
        "train",
        "evaluate",
        "optimize",
    }
    for task in containers:
        assert task.user == "1234:5678"
        assert task.environment["USER"] == "alice"  # MLflow's `mlflow.user` tag names the person
