"""The VisDrone-rule read: a tuning trial's ``best.pt`` scored under VisDrone's ignored-region rule.

It measures, never selects. Each trial predicts val once at its imgsz with val's floor and cap,
and the same predictions are scored twice: under VisDrone's rule (``val-visdrone-rule/`` keys,
the labeling trials' scale) and under the repo's (``val/``). Each trial is one
``<trial>-val-visdrone-rule`` run in the labeling trials' experiment, tagged
``tune.stage=visdrone-rule-read``.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from mlops_cv.startup import exits_on_startup_error
from mlops_cv.tuning.autolabel_visdrone_vid import visdrone_vid

logger = logging.getLogger(__name__)

SPLIT = "val"


def both_rules(
    detections: Mapping[str, np.ndarray],
    images: list[visdrone_vid.Image],
    rule_yaml: Path,
    repo_yaml: Path,
) -> dict[str, float]:
    """Val readings under VisDrone's rule and, on the same detections, under the repo's."""
    rule = visdrone_vid.drop_all_ignored(detections, images)
    return visdrone_vid.readings(
        rule, rule_yaml, SPLIT, visdrone_vid.visdrone_rule_split(SPLIT)
    ) | visdrone_vid.readings(detections, repo_yaml, SPLIT, SPLIT)


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    from mlops_cv.tuning.autolabel_visdrone_vid.label_eval import AUTOLABEL_EXPERIMENT
    from mlops_cv.tuning.common import CONF, MAX_DET, finished_trial

    p = argparse.ArgumentParser(description="Score tuning trials' best.pt under VisDrone's rule.")
    p.add_argument("trials", nargs="+", help="trial run names, e.g. stage2-s1280")
    args = p.parse_args(argv)

    os.environ["MLFLOW__EXPERIMENT"] = (
        AUTOLABEL_EXPERIMENT  # the trials stay in the tune experiment
    )
    from mlops_cv.config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")

    from ultralytics import YOLO

    from mlops_cv.data.subset import IMAGES_DIRNAME, dataset_yaml
    from mlops_cv.orchestration.handoff import best_weights_uri
    from mlops_cv.tracking import client
    from mlops_cv.tracking.resolve import resolve_model
    from mlops_cv.tuning.yolo26_visdrone_vid.zero_shot import result_rows

    mlflow = client.connect(settings)
    subset_dir = settings.data.subset_dir
    images = visdrone_vid.read_split(settings.data.raw_dir, subset_dir, SPLIT)
    rule_yaml, repo_yaml = visdrone_vid.rule_yaml(settings, SPLIT), dataset_yaml(settings)

    for trial in args.trials:
        run = finished_trial(mlflow, trial)
        imgsz = int(run.data.params["imgsz"])
        weights, _ = resolve_model(best_weights_uri(run.info.run_id))
        detections = {
            Path(r.path).name: result_rows(r)
            for r in YOLO(str(weights)).predict(
                str(subset_dir / IMAGES_DIRNAME / SPLIT),  # a directory streams frame by frame
                imgsz=imgsz,
                device=settings.training.device,
                conf=CONF,
                max_det=MAX_DET,
                stream=True,
                verbose=False,
            )
        }
        readings = both_rules(detections, images, rule_yaml, repo_yaml)
        name = f"{trial}-{visdrone_vid.visdrone_rule_split(SPLIT)}"
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("tune.stage", "visdrone-rule-read")
            run_params = {"trial": trial, "trial_run_id": run.info.run_id, "imgsz": imgsz}
            mlflow.log_params(run_params | {"conf": CONF, "max_det": MAX_DET})
            mlflow.log_metrics(readings)
        logger.info(f"{name}: {readings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
