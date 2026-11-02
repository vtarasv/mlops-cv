"""Run one tuning stage: ``configs/tuning/<case>/<stage>.yaml`` -> one training process per trial.

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
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import numpy as np
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

# val's own floor and box cap (``model.val`` defaults): detections scored here are predicted with
# them. For a class-mapped model, the cap applies before the mapping drops classes, as in the model.
CONF = 0.001
MAX_DET = 300


def load_stage(path: Path) -> dict[str, dict[str, Any]]:
    """``{run name: knobs}``, each trial's knobs over the stage's ``defaults``.

    A trial's section (a mapping) updates the default's keys, one level deep; any value inside
    one replaces the default's whole.
    """
    stage = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = stage.get("defaults", {})
    trials: dict[str, dict[str, Any]] = {}
    for entry in stage["trials"]:
        knobs = {**defaults}
        for key, value in entry.items():
            base = knobs.get(key)
            knobs[key] = {**base, **value} if isinstance(base, dict) else value
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


def finished_trial(mlflow: Any, name: str) -> Any:
    """The latest finished run named ``name`` in the tune experiment."""
    runs = mlflow.search_runs(
        experiment_names=[TUNE_EXPERIMENT],
        filter_string=f"attributes.run_name = '{name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise ValueError(f"no finished trial run named {name}")
    return runs[0]


def readings(
    detections: Mapping[str, np.ndarray],
    data_yaml: Path,
    split: str,
    prefix: str,
    frames: Collection[str] | None = None,
) -> dict[str, float]:
    """Merged detections scored on ``split`` (or its ``frames``) -> trial keys under ``prefix``."""
    from mlops_cv.evaluation.score_detections import score_with_ap_small
    from mlops_cv.tracking.metric_keys import (
        MAP_SMALL,
        PRIMARY,
        headline_metrics,
        metric_key,
        per_class_metrics,
    )

    metrics, ap_s = score_with_ap_small(detections, data_yaml, split, frames)
    return (
        headline_metrics(metrics.box, prefix=prefix)
        | {metric_key(prefix, MAP_SMALL): ap_s}
        | per_class_metrics(
            metrics.maps,
            metrics.ap_class_index,
            metrics.names,
            prefix=metric_key(prefix, PRIMARY),
        )
    )


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
    p.add_argument("stage", type=Path, help="configs/tuning/<case>/<stage>.yaml")
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
