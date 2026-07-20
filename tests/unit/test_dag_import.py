"""DagBag import guard: every DAG under ``dags/`` parses cleanly with the expected topology.
Runs in CI (the ``airflow`` uv group is CI-synced).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Importing airflow materializes AIRFLOW_HOME (default ~/airflow) as a side effect — point it at
# a throwaway BEFORE the importorskip below.
os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp(prefix="airflow-home-"))
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")

airflow = pytest.importorskip("airflow")

DAGS_DIR = Path(__file__).resolve().parents[2] / "dags"
CT_DAG = "aerial_object_detection_ct"


@pytest.fixture(scope="module")
def dagbag():
    from airflow.dag_processing.dagbag import DagBag

    # The CT DAG requires these at parse time (no in-DAG fallbacks). Set them only while the DagBag
    # parses and revert on return, so they don't leak into other test modules' os.environ.
    with pytest.MonkeyPatch.context() as mp:
        repo = Path(__file__).resolve().parents[2]
        mp.setenv("HOST_SUBSET_DIR", str(repo / "data" / "subset"))
        mp.setenv("HOST_WEIGHTS", str(repo / "data" / "weights" / "yolo26s.pt"))
        mp.setenv("TRAIN_IMAGE", "mlops-cv-train:test")
        mp.setenv("MLFLOW__TRACKING_URI", "http://mlflow:5000")
        return DagBag(dag_folder=str(DAGS_DIR), include_examples=False)


def test_dags_import_cleanly(dagbag) -> None:
    assert dagbag.import_errors == {}
    assert set(dagbag.dag_ids) == {CT_DAG, "smoke_test"}


def test_ct_dag_topology(dagbag) -> None:
    ct = dagbag.dags[CT_DAG]  # the parsed in-memory DAG
    assert {t.task_id for t in ct.tasks} == {
        "validate_data",
        "train",
        "parse_train_output",
        "evaluate",
        "gate",
        "promote",
        "skip_promotion",
    }
    down = {t.task_id: set(t.downstream_task_ids) for t in ct.tasks}
    assert down["validate_data"] == {"train"}
    assert down["train"] == {"parse_train_output"}
    assert down["parse_train_output"] == {"evaluate", "promote"}  # promote templates the version
    assert down["evaluate"] == {"gate"}
    assert down["gate"] == {"promote", "skip_promotion"}  # the champion/challenger branch


def test_ct_dag_run_policy(dagbag) -> None:
    ct = dagbag.dags[CT_DAG]
    assert ct.catchup is False  # a missed week must not queue a backfill of GPU trainings
    assert ct.max_active_runs == 1  # one CT cycle at a time
    assert ct.params["epochs"] == 3
