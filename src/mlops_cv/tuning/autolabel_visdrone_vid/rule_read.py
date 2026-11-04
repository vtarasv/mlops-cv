"""The VisDrone-rule read: a tuning trial's ``best.pt`` scored under VisDrone's ignored-region rule.

It measures, never selects. Each trial predicts the split (val, or test once) at its imgsz with
val's floor and cap, and the same predictions are scored twice: under VisDrone's rule
(``<split>-visdrone-rule/`` keys, the labeling trials' scale) and under the repo's (``<split>/``).
Each trial is one ``<trial>-<split>-visdrone-rule`` run in the labeling trials' experiment, tagged
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


def both_rules(
    detections: Mapping[str, np.ndarray],
    images: list[visdrone_vid.Image],
    rule_yaml: Path,
    repo_yaml: Path,
    split: str,
) -> dict[str, float]:
    """``split``'s readings under VisDrone's rule and, on the same detections, under the repo's."""
    rule = visdrone_vid.drop_all_ignored(detections, images)
    return visdrone_vid.readings(
        rule, rule_yaml, split, visdrone_vid.visdrone_rule_split(split)
    ) | visdrone_vid.readings(detections, repo_yaml, split, split)


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    from mlops_cv.tuning import yolo26_visdrone_vid
    from mlops_cv.tuning.autolabel_visdrone_vid import EXPERIMENT
    from mlops_cv.tuning.common import CONF, MAX_DET, finished_trial

    p = argparse.ArgumentParser(description="Score tuning trials' best.pt under VisDrone's rule.")
    p.add_argument("trials", nargs="+", help="trial run names, e.g. stage2-s1280")
    p.add_argument(
        "--split", choices=visdrone_vid.RULE_SPLITS, default="val", help="test is read once"
    )
    args = p.parse_args(argv)

    os.environ["MLFLOW__EXPERIMENT"] = EXPERIMENT  # the trials stay in YOLO tuning's
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
    split = args.split
    images = visdrone_vid.read_split(settings.data.raw_dir, subset_dir, split)
    rule_yaml, repo_yaml = visdrone_vid.rule_yaml(settings, split), dataset_yaml(settings)

    for trial in args.trials:
        run = finished_trial(mlflow, yolo26_visdrone_vid.EXPERIMENT, trial)
        imgsz = int(run.data.params["imgsz"])
        weights, _ = resolve_model(best_weights_uri(run.info.run_id))
        detections = {
            Path(r.path).name: result_rows(r)
            for r in YOLO(str(weights)).predict(
                str(subset_dir / IMAGES_DIRNAME / split),  # a directory streams frame by frame
                imgsz=imgsz,
                device=settings.training.device,
                conf=CONF,
                max_det=MAX_DET,
                stream=True,
                verbose=False,
            )
        }
        readings = both_rules(detections, images, rule_yaml, repo_yaml, split)
        name = f"{trial}-{visdrone_vid.visdrone_rule_split(split)}"
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("tune.stage", "visdrone-rule-read")
            run_params = {"trial": trial, "trial_run_id": run.info.run_id, "imgsz": imgsz}
            mlflow.log_params(run_params | {"conf": CONF, "max_det": MAX_DET})
            mlflow.log_metrics(readings)
        logger.info(f"{name}: {readings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
