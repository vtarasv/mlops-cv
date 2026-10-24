"""Continuous-training DAGs, one per deployment target (server, edge): ensure data -> profile ->
validate -> train -> evaluate -> gate -> optimize -> promote/skip.

Thin orchestration over ``mlops_cv``: decisions run in-process (pure code), the heavy work runs
in ``DockerOperator`` containers on the shared platform network — GPU train/eval in the train
image, CPU data prep (Beam ingestion + dataset profiling) in the beam image. Fires weekly or on
a ``new-training-data`` asset event (POSTed by an external producer, e.g. a drift monitor).

Container contract (owned by ``mlops_cv.orchestration.handoff`` — both ends import it):
the DAG builds each container's command there, ``train`` replies with a ``TrainHandoff``
line (its last stdout line = XCom), and ``evaluate`` logs test metrics + report + visuals
onto the same training run and replies with the ``GateVerdict`` line the branch decides
on. On a gate pass the challenger's serving variants are built and benchmarked (``optimize``),
then it is promoted to the target's ``champion`` registry alias — only a servable version is.
The GPU tasks of both DAGs share the 1-slot ``gpu`` pool: two trainings at once spill label
assignment onto the CPU.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import Asset, Param, dag, task
from docker.types import DeviceRequest, Mount

from mlops_cv.data.subset import demo_store_present
from mlops_cv.data.validate import DatasetValidationError, validate_dataset
from mlops_cv.optimize.variants import EDGE, HEADS, ONE_TO_MANY, SERVER, TARGETS
from mlops_cv.orchestration.handoff import (
    GateVerdict,
    TrainHandoff,
    evaluate_cmd,
    ingest_cmd,
    optimize_cmd,
    profile_cmd,
    train_cmd,
)
from mlops_cv.pipelines.profiling import is_profile_current

# pyright: reportUnusedExpression=false

logger = logging.getLogger(__name__)

# Orchestration wiring is injected by the Airflow stack (docker-compose/.env.airflow + Makefile).
HOST_RAW_DIR = os.environ.get("HOST_RAW_DIR", "")  # optional: only the ingest task reads raw
SUBSET_DIR = os.environ["HOST_SUBSET_DIR"]
WEIGHTS = os.environ["HOST_WEIGHTS"]  # pretrained-weights cache: downloaded once, reused
BEAM_IMAGE = os.environ["BEAM_IMAGE"]
TRAIN_IMAGE = os.environ["TRAIN_IMAGE"]
OPTIMIZE_IMAGE = os.environ["OPTIMIZE_IMAGE"]
MLFLOW_URI = os.environ["MLFLOW__TRACKING_URI"]
HOST_USER = os.environ["HOST_USER"]  # uid:gid, so bind-mounted outputs stay host-user-owned
HOST_USERNAME = os.environ["HOST_USERNAME"]

# Host paths are bind-mounted at identical container paths ("path parity") so tasks and the
# scheduler-side checks share one subset path, the one each task stamps into its dataset YAML.
WEIGHTS_DIR = str(Path(WEIGHTS).parent)  # mount the dir: the file only exists after 1st download

NEW_DATA_ASSET = Asset("new-training-data")

# Each target's default YOLO26 size (training's TARGET_MODELS, pinned by the DAG tests).
DEFAULT_SIZE = {SERVER: "m", EDGE: "n"}

# Joins after a branch: run when nothing failed and at least one upstream path was followed.
JOIN_RULE = "none_failed_min_one_success"

_TASK_ENV = {
    "MLFLOW__TRACKING_URI": MLFLOW_URI,
    "DATA__SUBSET_DIR": SUBSET_DIR,
    "TRAINING__WEIGHTS": WEIGHTS,
    "USER": HOST_USERNAME,  # getpass reads it before passwd, where the host uid has no entry
}
# rw: the ingest/profile tasks write the dataset here; ultralytics writes its labels/*.cache.
_MOUNTS = [
    Mount(source=SUBSET_DIR, target=SUBSET_DIR, type="bind"),
    Mount(source=WEIGHTS_DIR, target=WEIGHTS_DIR, type="bind"),
]
# Raw data is needed ONLY by the ingest task (subset rebuild + demo store materialization).
_INGEST_MOUNTS = _MOUNTS + (
    [Mount(source=HOST_RAW_DIR, target=HOST_RAW_DIR, type="bind", read_only=True)]
    if HOST_RAW_DIR
    else []
)

_DOCKER_BASE = {
    "docker_url": "unix://var/run/docker.sock",
    "user": HOST_USER,
    "network_mode": "mlops-cv-network",  # reach the MLflow server as http://mlflow:5000
    "mounts": _MOUNTS,
    "auto_remove": "success",  # keep failed containers around for debugging
    "mount_tmp_dir": False,  # the default host tmp mount breaks for socket-launched siblings
}

_GPU_DOCKER_COMMON = {
    **_DOCKER_BASE,
    "image": TRAIN_IMAGE,
    "environment": _TASK_ENV,
    # The programmatic `--gpus all`.
    "device_requests": [DeviceRequest(count=-1, capabilities=[["gpu"]])],
    # Docker's default /dev/shm is 64 MB; torch DataLoader workers share tensors through it.
    "shm_size": 2 * 1024**3,
    "pool": "gpu",  # 1 slot, created by airflow-init: one GPU task at a time across both DAGs
    # A hung GPU container would otherwise hold the gpu pool slot forever.
    "execution_timeout": timedelta(hours=2),
    "do_xcom_push": True,  # XCom = the container's last stdout line
}

_OPTIMIZE_DOCKER = {
    **_GPU_DOCKER_COMMON,
    "image": OPTIMIZE_IMAGE,
    "do_xcom_push": False,
}

# CPU data-prep siblings (beam image): no GPU, no XCom, default /dev/shm.
_CPU_DOCKER_COMMON = {
    **_DOCKER_BASE,
    "image": BEAM_IMAGE,
    "environment": _TASK_ENV,
    "execution_timeout": timedelta(hours=1),
    "do_xcom_push": False,
}


def _params(target: str) -> dict[str, Param]:
    """The trigger form: static defaults per target (a param default cannot follow another)."""
    return {
        "model": Param(DEFAULT_SIZE[target], type="string", enum=["n", "s", "m"]),
        "imgsz": Param(1280, type="integer", enum=[640, 960, 1280], description="train + export"),
        "head": Param(
            ONE_TO_MANY,
            type="string",
            enum=list(HEADS),
            description="scored and exported; edge's NCNN always runs one-to-many",
        ),
        "epochs": Param(30, type="integer", minimum=1, description="3 for a demo-short run"),
        "batch": Param(
            -1, type="integer", minimum=-1, description="-1 = autobatch; pin it if a run OOMs"
        ),
    }


def ct_dag(target: str) -> None:
    """One CT DAG training, gating and promoting into ``target``'s registered model."""

    @dag(
        dag_id=f"aerial_object_detection_ct_{target}",
        description=f"Continuous training for the {target} target: data, train, gate, promote.",
        # Weekly runs off for now (asset event or manual trigger only); to restore:
        # schedule=AssetOrTimeSchedule(
        #     timetable=CronTriggerTimetable("0 3 * * 1", timezone="UTC"),  # Mon 03:00 UTC
        #     assets=NEW_DATA_ASSET,
        # ),
        schedule=[NEW_DATA_ASSET],
        start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
        catchup=False,
        max_active_runs=1,  # one CT cycle at a time; raise in outside of local-dev
        params=_params(target),
        tags=["continuous-training", "yolo", "mlflow", target],
    )
    def aerial_object_detection_ct() -> None:
        @task.branch
        def check_subset() -> str:
            """Self-heal the subset: rebuild (with its demo store) from raw when missing/invalid."""
            try:
                report = validate_dataset(SUBSET_DIR)
            except (DatasetValidationError, FileNotFoundError) as exc:
                logger.warning(f"subset missing/invalid ({exc}) -> rebuilding from raw")
                return "build_subset"
            logger.info(report.summary())
            if not demo_store_present(SUBSET_DIR):
                if HOST_RAW_DIR:
                    logger.warning("demo store missing -> rebuilding the subset from raw")
                    return "build_subset"
                logger.warning("demo store missing and no HOST_RAW_DIR; evaluate will skip demos")
            return "check_profiled"

        # The Airflow->Beam handoff.
        build_subset = DockerOperator(
            task_id="build_subset",
            command=ingest_cmd(raw_dir=HOST_RAW_DIR, output_dir=SUBSET_DIR),
            **{**_CPU_DOCKER_COMMON, "mounts": _INGEST_MOUNTS},
        )

        @task.branch(trigger_rule=JOIN_RULE)
        def check_profiled() -> str:
            """Skip re-profiling when the profile is current (stamp = subset manifest + params)."""
            if is_profile_current(SUBSET_DIR):
                logger.info(f"dataset profile at {SUBSET_DIR} is up to date; skipping profile")
                return "validate_data"
            return "profile"

        profile = DockerOperator(
            task_id="profile",
            command=profile_cmd(input_dir=SUBSET_DIR),
            **_CPU_DOCKER_COMMON,
        )

        @task(trigger_rule=JOIN_RULE)
        def validate_data() -> str:
            """Fail fast on schema/value skew before spending GPU time (raises on any error)."""
            report = validate_dataset(SUBSET_DIR)
            logger.info(report.summary())
            return report.summary()

        train = DockerOperator(
            task_id="train",
            command=train_cmd(
                target=target,
                model="{{ params.model }}",
                imgsz="{{ params.imgsz }}",
                epochs="{{ params.epochs }}",
                batch="{{ params.batch }}",
            ),
            # m@1280 x 30 epochs on the subset took ~1 GPU-hour.
            **{**_GPU_DOCKER_COMMON, "execution_timeout": timedelta(hours=3)},
        )

        @task
        def parse_train_output(raw: str) -> dict[str, str]:
            """Parse train's handoff line into the XCom dict downstream tasks template against."""
            return TrainHandoff.parse(raw).model_dump()

        _run_id = "{{ ti.xcom_pull(task_ids='parse_train_output')['run_id'] }}"
        _head = "{{ params.head }}"
        evaluate = DockerOperator(
            task_id="evaluate",
            command=evaluate_cmd(run_id=_run_id, target=target, head=_head),
            **_GPU_DOCKER_COMMON,
        )

        @task.branch
        def gate(verdict_line: str) -> str:
            """Champion/challenger decision on evaluate's verdict — pure parsing, no MLflow."""
            verdict = GateVerdict.parse(verdict_line)
            logger.info(
                f"gate: passed={verdict.passed} candidate={verdict.candidate_primary} "
                f"champion={verdict.champion_primary}"
            )
            return "optimize" if verdict.passed else "skip_promotion"

        optimize = DockerOperator(
            task_id="optimize",
            command=optimize_cmd(run_id=_run_id, target=target, head=_head),
            **_OPTIMIZE_DOCKER,
        )

        @task
        def promote(train_info: dict[str, str]) -> None:
            """Point the target's champion alias at the challenger (gate passed, variants built)."""
            from mlops_cv.config import get_settings
            from mlops_cv.evaluation.gate import promote as set_champion_alias

            set_champion_alias(get_settings(), train_info["version"], target)

        subset_choice = check_subset()
        profiled_choice = check_profiled()
        validated = validate_data()
        info = parse_train_output(train.output)  # type: ignore[arg-type]
        verdict = gate(evaluate.output)  # type: ignore[arg-type]
        promoted = promote(info)  # type: ignore[arg-type]
        skipped = EmptyOperator(task_id="skip_promotion")

        subset_choice >> [build_subset, profiled_choice]
        build_subset >> profiled_choice
        profiled_choice >> [profile, validated]
        profile >> validated
        validated >> train
        info >> evaluate
        verdict >> [optimize, skipped]
        optimize >> promoted

    aerial_object_detection_ct()


for _target in TARGETS:
    ct_dag(_target)
