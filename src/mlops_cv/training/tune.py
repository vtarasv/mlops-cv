"""Run one tuning-grid stage: ``configs/tuning/<stage>.yaml`` -> one training process per trial.

Trials train one at a time, never register, and log to the tune experiment. Each trial's
latency is measured afterwards, alone on the GPU, onto its run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from mlops_cv.config import get_settings
from mlops_cv.data.subset import IMAGES_DIRNAME
from mlops_cv.orchestration.handoff import best_weights_uri
from mlops_cv.startup import exits_on_startup_error
from mlops_cv.training.train import TRIAL_SPLIT

logger = logging.getLogger(__name__)

TUNE_EXPERIMENT = "aerial-object-detection-tune"

# Knobs with a named training flag (headed for Airflow); every other knob goes through ``--set``.
FLAGS = ("model", "imgsz", "epochs", "batch", "seed")


def load_stage(path: Path) -> dict[str, dict[str, Any]]:
    """``{run name: knobs}``, each trial's knobs over the stage's ``defaults``."""
    stage = yaml.safe_load(path.read_text(encoding="utf-8"))
    trials: dict[str, dict[str, Any]] = {}
    for entry in stage["trials"]:
        knobs = {**stage.get("defaults", {}), **entry}
        name = f"{path.stem}-{knobs.pop('name')}"
        if name in trials:
            raise ValueError(f"trial {name} repeats in {path}")
        trials[name] = knobs
    return trials


def trial_argv(name: str, knobs: Mapping[str, Any]) -> list[str]:
    """The training command line for one trial; ``--set`` values travel as JSON (valid YAML)."""
    argv = [sys.executable, "-m", "mlops_cv.training", "--run-name", name, "--no-register"]
    for key, value in knobs.items():
        if key in FLAGS:
            argv += [f"--{key}", str(value)]
        else:
            argv += ["--set", f"{key}={json.dumps(value)}"]
    return argv


def _record_latency(mlflow: Any, name: str, imgsz: int, since_ms: int) -> None:
    """Benchmark the trial's logged ``best.pt`` and log the latency onto its run."""
    from ultralytics import YOLO

    from mlops_cv.evaluation.evaluate import latency_stub
    from mlops_cv.tracking.resolve import resolve_model

    settings = get_settings()
    run = mlflow.search_runs(
        experiment_names=[TUNE_EXPERIMENT],
        filter_string=f"attributes.run_name = '{name}' and attributes.start_time >= {since_ms}",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )[0]
    weights, _ = resolve_model(best_weights_uri(run.info.run_id))
    latency = latency_stub(
        YOLO(str(weights)),
        settings.data.subset_dir / IMAGES_DIRNAME / TRIAL_SPLIT,
        imgsz=imgsz,
        device=settings.training.device,
    )
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_metrics(latency)


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run one tuning-grid stage, one process per trial.")
    p.add_argument("stage", type=Path, help="configs/tuning/<stage>.yaml")
    args = p.parse_args(argv)

    os.environ["MLFLOW__EXPERIMENT"] = TUNE_EXPERIMENT  # trials inherit it: never production's
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    trials = load_stage(args.stage)

    since_ms = int(time.time() * 1000)
    failed = [
        name for name, knobs in trials.items() if subprocess.run(trial_argv(name, knobs)).returncode
    ]

    from mlops_cv.tracking import client

    mlflow = client.connect(settings)
    for name, knobs in trials.items():
        if name not in failed:
            _record_latency(mlflow, name, knobs.get("imgsz", settings.training.imgsz), since_ms)

    if failed:
        logger.error(f"{len(failed)}/{len(trials)} trials failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
