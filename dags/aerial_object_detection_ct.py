"""Continuous-training DAG: validate data -> train -> evaluate -> gate -> promote/skip.

Thin orchestration over ``mlops_cv``: decisions run in-process (pure code), the GPU work runs in
``DockerOperator`` containers on the shared platform network. Fires weekly or on a
``new-training-data`` asset event (POSTed by an external producer, e.g. a drift monitor).

Container contract: ``train`` prints ``{"version", "run_id"}`` as its last stdout line (XCom);
``evaluate --run-id --exit-zero`` logs test metrics + report + visuals onto the same training run
and prints the gate-verdict JSON the branch decides on. The challenger is promoted to the
``champion`` registry alias only on a gate pass.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import timedelta

import pendulum
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import Asset, dag, task
from airflow.timetables.assets import AssetOrTimeSchedule
from airflow.timetables.trigger import CronTriggerTimetable
from docker.types import DeviceRequest, Mount

from mlops_cv.data.validate import validate_dataset

logger = logging.getLogger(__name__)

# Orchestration wiring is injected by the Airflow stack (docker-compose/.env.airflow + Makefile).
HOST_PROJECT_DIR = os.environ["HOST_PROJECT_DIR"]
HOST_RAW_DIR = os.environ.get("HOST_RAW_DIR", "")  # optional: demo clips render from raw frames
TRAIN_IMAGE = os.environ["TRAIN_IMAGE"]
MLFLOW_URI = os.environ["MLFLOW__TRACKING_URI"]

# Host paths are bind-mounted at identical container paths ("path parity") so the absolute
# `path:` stamped into the generated dataset YAML resolves inside the task containers too.
DATA_DIR = f"{HOST_PROJECT_DIR}/data"
SUBSET_DIR = f"{DATA_DIR}/visdrone-vid-small"

NEW_DATA_ASSET = Asset("new-training-data")

_TASK_ENV = {
    "MLFLOW__TRACKING_URI": MLFLOW_URI,
    "ENV": "{{ var.value.get('CONFIG_ENV', 'local') }}",
    "DATA__SUBSET_DIR": SUBSET_DIR,
    # Pretrained weights cache inside the data mount: downloaded once, reused across runs.
    "TRAINING__WEIGHTS": f"{DATA_DIR}/weights/yolo26s.pt",
}
# rw: ultralytics `cache=disk` writes .npy files next to the images; the weights cache lives here.
_MOUNTS = [Mount(source=DATA_DIR, target=DATA_DIR, type="bind")]
if HOST_RAW_DIR:
    _TASK_ENV["DATA__RAW_DIR"] = HOST_RAW_DIR
    _MOUNTS.append(Mount(source=HOST_RAW_DIR, target=HOST_RAW_DIR, type="bind", read_only=True))

_DOCKER_COMMON = {
    "image": TRAIN_IMAGE,
    "docker_url": "unix://var/run/docker.sock",
    "network_mode": "mlops-cv-network",  # reach the MLflow server as http://mlflow:5000
    "environment": _TASK_ENV,
    "mounts": _MOUNTS,
    # The programmatic `--gpus all`.
    "device_requests": [DeviceRequest(count=-1, capabilities=[["gpu"]])],
    # Docker's default /dev/shm is 64 MB; torch DataLoader workers share tensors through it.
    "shm_size": 2 * 1024**3,
    # A hung GPU container would otherwise hold the single run slot (max_active_runs=1) forever.
    "execution_timeout": timedelta(hours=2),
    "auto_remove": "success",  # keep failed containers around for debugging
    "mount_tmp_dir": False,  # the default host tmp mount breaks for socket-launched siblings
    "do_xcom_push": True,  # XCom = the container's last stdout line
}


@dag(
    dag_id="aerial_object_detection_ct",
    description="Continuous training: validate data, train a challenger, evaluate, gate, promote.",
    schedule=AssetOrTimeSchedule(
        timetable=CronTriggerTimetable("0 3 * * 1", timezone="UTC"),  # weekly, Monday 03:00 UTC
        assets=NEW_DATA_ASSET,  # type: ignore[arg-type]
    ),
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,  # one CT cycle at a time; raise in outside of local-dev
    params={"epochs": 3},  # demo-short; raise for a real retrain
    tags=["continuous-training", "yolo", "mlflow"],
)
def aerial_object_detection_ct() -> None:
    @task
    def validate_data() -> str:
        """Fail fast on schema/value skew before spending GPU time (raises on any error)."""
        report = validate_dataset(SUBSET_DIR)
        logger.info(report.summary())
        return report.summary()

    train = DockerOperator(
        task_id="train",
        command=["python", "-m", "mlops_cv.training.train", "--epochs", "{{ params.epochs }}"],
        **_DOCKER_COMMON,
    )

    @task
    def parse_train_output(raw: str) -> dict[str, str]:
        """Parse train's handoff line into the XCom dict downstream tasks template against."""
        info = json.loads(raw)
        return {"version": str(info["version"]), "run_id": str(info["run_id"])}

    _run_id = "{{ ti.xcom_pull(task_ids='parse_train_output')['run_id'] }}"
    evaluate = DockerOperator(
        task_id="evaluate",
        command=[
            "python",
            "-m",
            "mlops_cv.eval.evaluate",
            "--model",
            f"runs:/{_run_id}/weights/best.pt",
            "--run-id",
            _run_id,  # consolidate test metrics + report + visuals on the training run
            "--exit-zero",  # a challenger loss is a verdict for the branch, not a task failure
            "--batch",
            "8",
        ],
        **_DOCKER_COMMON,
    )

    @task.branch
    def gate(verdict_line: str) -> str:
        """Champion/challenger decision on evaluate's verdict — pure JSON, no MLflow."""
        verdict = json.loads(verdict_line)
        logger.info(
            "gate: passed=%s candidate=%s champion=%s",
            verdict["passed"],
            verdict.get("candidate_primary"),
            verdict.get("champion_primary"),
        )
        return "promote" if verdict["passed"] else "skip_promotion"

    @task
    def promote(train_info: dict[str, str]) -> None:
        """Point the champion alias at the challenger version (the gate already passed)."""
        from mlops_cv.config import get_settings
        from mlops_cv.eval.gate import promote as set_champion_alias
        from mlops_cv.tracking import client

        client.configure()
        settings = get_settings()
        alias = settings.mlflow.champion_alias
        set_champion_alias(settings.mlflow.registered_model, train_info["version"], alias)
        logger.info("promoted v%s -> alias %r", train_info["version"], alias)

    validated = validate_data()
    info = parse_train_output(train.output)  # type: ignore[arg-type]
    verdict = gate(evaluate.output)  # type: ignore[arg-type]
    promoted = promote(info)  # type: ignore[arg-type]
    skipped = EmptyOperator(task_id="skip_promotion")

    validated >> train
    info >> evaluate
    verdict >> [promoted, skipped]


aerial_object_detection_ct()
