"""Zero-shot baselines: COCO-pretrained yolo26n/s/m on val, scored like a trial.

COCO classes map onto the merged classes and every other COCO class is dropped. Detections are
scored through ``score_detections``, so the AP math stays ultralytics'. Each model x imgsz is one
run in the tune experiment, with the trial's metric keys and latency stub.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mlops_cv.data.convert_visdrone_vid import YOLO_NAMES
from mlops_cv.startup import exits_on_startup_error

logger = logging.getLogger(__name__)

COCO_TO_MERGED: dict[str, str] = {
    "person": "person",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "bicycle": "two-three-wheeler",
    "motorcycle": "two-three-wheeler",
}

# val's own floor and box cap (``model.val`` defaults). The cap applies before the mapping drops
# COCO classes, as it would inside the model.
CONF = 0.001
MAX_DET = 300

IMGSZ = (640, 1280)


def class_lut(names: Mapping[int, str], table: Mapping[str, str] = COCO_TO_MERGED) -> np.ndarray:
    """A model's class id -> merged class id through ``table`` (by name), ``-1`` if dropped."""
    merged = {name: i for i, name in YOLO_NAMES.items()}
    lut = np.full(max(names) + 1, -1)
    for i, name in names.items():
        if name in table:
            lut[i] = merged[table[name]]
    return lut


def merge_classes(rows: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """``(n, 6) x1 y1 x2 y2 conf cls`` model rows -> merged rows, highest conf first.

    YOLO26's end2end top-k can emit one box under several classes (car and truck), which become
    the same box twice in one merged class: only the highest-conf copy is kept.
    """
    cls = lut[rows[:, 5].astype(int)]
    rows = np.column_stack([rows[:, :5], cls])[cls >= 0]
    rows = rows[np.argsort(-rows[:, 4], kind="stable")]
    _, first = np.unique(rows[:, [0, 1, 2, 3, 5]], axis=0, return_index=True)
    return rows[np.sort(first)]


def result_rows(result: Any) -> np.ndarray:
    """An ultralytics predict result -> ``(n, 6) x1 y1 x2 y2 conf cls`` in source pixels."""
    return result.boxes.data.cpu().numpy()


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


@exits_on_startup_error
def main() -> int:
    from mlops_cv.training.train import MODELS, TRIAL_SPLIT
    from mlops_cv.training.tune import TUNE_EXPERIMENT

    os.environ["MLFLOW__EXPERIMENT"] = TUNE_EXPERIMENT  # never production's experiment
    from mlops_cv.config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")

    from ultralytics import YOLO

    from mlops_cv.data.subset import IMAGES_DIRNAME, dataset_yaml
    from mlops_cv.evaluation.evaluate import latency_stub
    from mlops_cv.tracking import client

    mlflow = client.connect(settings)
    data_yaml = dataset_yaml(settings)
    images_dir = settings.data.subset_dir / IMAGES_DIRNAME / TRIAL_SPLIT
    device = settings.training.device

    for name in MODELS:
        model = YOLO(Path(settings.training.weights).with_name(f"{name}.pt"))
        lut = class_lut(model.names)
        for imgsz in IMGSZ:
            run_name = f"zero-shot-{name.removeprefix('yolo26')}{imgsz}"
            detections = {
                Path(r.path).name: merge_classes(result_rows(r), lut)
                for r in model.predict(
                    str(images_dir),
                    imgsz=imgsz,
                    device=device,
                    conf=CONF,
                    max_det=MAX_DET,
                    stream=True,
                    verbose=False,
                )
            }
            scores = readings(detections, data_yaml, TRIAL_SPLIT, TRIAL_SPLIT)
            scores |= latency_stub(model, images_dir, imgsz=imgsz, device=device)
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("tune.stage", "zero-shot")
                mlflow.log_params({"model": name, "imgsz": imgsz, "conf": CONF, "max_det": MAX_DET})
                mlflow.log_metrics(scores)
            logger.info(f"{run_name}: {scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
