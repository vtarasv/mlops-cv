"""The test read: a tuning trial's ``best.pt`` scored on test with the served end2end head.

It measures, never selects. Each trial is one ``<trial>-test`` run in the tune experiment
(``tune.stage=test-read``) holding ``test/`` keys that mirror the trial's ``val/`` ones. Latency
does not depend on the split, so the trial's own readings stand.
"""

from __future__ import annotations

import argparse
import logging
import os

from mlops_cv.startup import exits_on_startup_error

logger = logging.getLogger(__name__)

SPLIT = "test"


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    from mlops_cv.training.tune import TUNE_EXPERIMENT, finished_trial

    p = argparse.ArgumentParser(description="Score tuning trials' best.pt on test.")
    p.add_argument("trials", nargs="+", help="trial run names, e.g. stage2-n1280")
    args = p.parse_args(argv)

    os.environ["MLFLOW__EXPERIMENT"] = TUNE_EXPERIMENT  # never production's experiment
    from mlops_cv.config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")

    from ultralytics import YOLO

    from mlops_cv.data.subset import dataset_yaml
    from mlops_cv.evaluation.evaluate import split_metrics
    from mlops_cv.orchestration.handoff import best_weights_uri
    from mlops_cv.tracking import client
    from mlops_cv.tracking.resolve import resolve_model

    mlflow = client.connect(settings)
    data_yaml = dataset_yaml(settings)

    for trial in args.trials:
        run = finished_trial(mlflow, trial)
        imgsz = int(run.data.params["imgsz"])
        weights, _ = resolve_model(best_weights_uri(run.info.run_id))
        name = f"{trial}-{SPLIT}"
        readings = split_metrics(
            YOLO(str(weights)),
            data_yaml,
            SPLIT,
            imgsz=imgsz,
            device=settings.training.device,
            batch=8,  # the trial's own val batch
            name=name,
        )
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("tune.stage", "test-read")
            mlflow.log_params({"trial": trial, "trial_run_id": run.info.run_id, "imgsz": imgsz})
            mlflow.log_metrics(readings)
        logger.info(f"{name}: {readings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
