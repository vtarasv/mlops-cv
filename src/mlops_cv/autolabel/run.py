"""Run one labeling stage: ``configs/autolabel/<stage>.yaml`` -> one MLflow run per labeling trial.

Each trial labels a whole VID scoring set. The filtered detector boxes (``det``) and the
mask-derived boxes (``seg``) are scored apart, through ultralytics' validator, after the
ignored-region drop. The auto-labels are logged as one COCO RLE file per trial and set.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import tempfile
from collections import Counter
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mlops_cv.autolabel import visdrone_vid
from mlops_cv.autolabel.labeler import CHAMPION, Labeler, LabelingConfig, Labels, read_rgb
from mlops_cv.startup import exits_on_startup_error

logger = logging.getLogger(__name__)

AUTOLABEL_EXPERIMENT = "aerial-auto-label"
# SAM 3's per-prompt cap. Every row is also scored on the images where no raw category has
# more ground truth than this, so a capped row compares on the images its cap can't bind.
SUBSET_CAP = 200
# Images left out of the latency readings.
WARMUP = 3
DET, SEG = "det", "seg"
SET_TAG = "autolabel.set"


def autolabels_file(scoring_set: str) -> str:
    """The trial's artifact: its auto-labels as COCO RLE."""
    return f"autolabels-{scoring_set}.json"


def load_stage(path: Path) -> dict[str, LabelingConfig]:
    """``{run name: labeling config}``, each trial's sections over the stage's ``defaults``.

    A trial's section (``detector``, ``box_filter``) updates the default's keys; any value inside
    one, ``prompts`` included, replaces the default's whole.
    """
    stage = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = stage.get("defaults", {})
    trials: dict[str, LabelingConfig] = {}
    for entry in stage["trials"]:
        entry = dict(entry)
        name = f"{path.stem}-{entry.pop('name')}"
        if name in trials:
            raise ValueError(f"trial {name} repeats in {path}")
        merged = {**defaults}
        for key, value in entry.items():
            base = merged.get(key)
            merged[key] = {**base, **value} if isinstance(base, dict) else value
        trials[name] = LabelingConfig.model_validate(merged)
    return trials


def params(config: LabelingConfig) -> dict[str, Any]:
    """The config flattened to ``section.key`` MLflow params."""
    return {
        f"{key}.{k}" if isinstance(value, dict) else key: v
        for key, value in config.model_dump().items()
        for k, v in (value.items() if isinstance(value, dict) else [(key, value)])
    }


def mask_rows(labels: Labels) -> np.ndarray:
    """The mask-derived boxes with their detection's conf and class; empty masks are dropped."""
    from pycocotools import mask

    mask_utils: Any = mask

    if not labels.masks:
        return np.zeros((0, 6), np.float32)
    xywh = mask_utils.toBbox(labels.masks)
    keep = mask_utils.area(labels.masks) > 0
    rows = np.column_stack([xywh[:, :2], xywh[:, :2] + xywh[:, 2:], labels.rows[:, 4:6]])
    return rows[keep]


def outside_ignored(rows: np.ndarray, image: visdrone_vid.Image) -> np.ndarray:
    """``rows`` (x1 y1 x2 y2 ...) without those the toolkit's ignored-region rule drops."""
    xywh = np.column_stack([rows[:, :2], rows[:, 2:4] - rows[:, :2]])
    return rows[visdrone_vid.outside_ignored(xywh, image.ignored, image.width, image.height)]


def uncapped(images: list[visdrone_vid.Image], cap: int) -> list[str]:
    """The images where no raw category has more than ``cap`` ground-truth boxes."""
    return [
        im.name for im in images if max(Counter(im.categories.tolist()).values(), default=0) <= cap
    ]


def recall_ceiling(images: list[visdrone_vid.Image], cap: int) -> float:
    """The recall left when each raw category keeps at most ``cap`` boxes per image."""
    counts = [n for im in images for n in Counter(im.categories.tolist()).values()]
    total = sum(counts)
    return 1 - sum(max(0, n - cap) for n in counts) / total if total else 1.0


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def readings(
    detections: Mapping[str, np.ndarray],
    data_yaml: Path,
    prefix: str,
    frames: Collection[str] | None,
) -> dict[str, float]:
    """The validator's keys under ``prefix``, plus F1 at IoU 0.5 from its precision and recall."""
    from mlops_cv.evaluation.score_detections import readings as validator_readings
    from mlops_cv.tracking.metric_keys import F1, metric_key

    out = validator_readings(detections, data_yaml, visdrone_vid.SCORING_SPLIT, prefix, frames)
    p, r = out[metric_key(prefix, "precision")], out[metric_key(prefix, "recall")]
    return out | {metric_key(prefix, F1): f1(p, r)}


def scores(
    images: list[visdrone_vid.Image], labels: list[Labels], data_yaml: Path, *, full: bool
) -> dict[str, float]:
    """Each box kind scored on every image, then on the ``SUBSET_CAP`` subset."""
    from mlops_cv.tracking.metric_keys import metric_key

    kinds = {DET: [lab.rows for lab in labels]}
    if labels and labels[0].masks is not None:
        kinds[SEG] = [mask_rows(lab) for lab in labels]
    frames = None if full else {im.name for im in images}
    subset = set(uncapped(images, SUBSET_CAP))
    out: dict[str, float] = {}
    for kind, rows in kinds.items():
        detections = {im.name: outside_ignored(r, im) for im, r in zip(images, rows, strict=True)}
        out |= readings(detections, data_yaml, kind, frames)
        out |= readings(detections, data_yaml, f"{kind}/uncapped{SUBSET_CAP}", subset)
        out[metric_key(kind, "boxes_per_image")] = float(np.mean([len(r) for r in rows]))
    out[metric_key(DET, "detected_per_image")] = float(np.mean([lab.detected for lab in labels]))
    cap = f"gt/cap{SUBSET_CAP}"
    out[metric_key(cap, "recall_ceiling")] = recall_ceiling(images, SUBSET_CAP)
    out[metric_key(cap, "uncapped_images")] = float(len(subset))
    return out


def latency(labels: list[Labels]) -> dict[str, float]:
    """p50 and mean wall-clock per stage and in total, warm-up images left out."""
    from mlops_cv.tracking.metric_keys import LATENCY_PREFIX, metric_key

    timed = labels[WARMUP:] or labels
    stages = {s: np.array([lab.ms[s] for lab in timed]) for s in timed[0].ms}
    stages["total"] = np.sum(list(stages.values()), axis=0)
    out: dict[str, float] = {}
    for stage, ms in stages.items():
        prefix = metric_key(LATENCY_PREFIX, stage)
        out[metric_key(prefix, "p50_ms")] = float(np.median(ms))
        out[metric_key(prefix, "mean_ms")] = float(ms.mean())
    return out


def coco_autolabels(images: list[visdrone_vid.Image], labels: list[Labels]) -> dict[str, Any]:
    """The trial's COCO file: ``bbox`` is the mask's box, the detector's goes in ``det_bbox``.

    An empty mask is no auto-label. Without a segmenter, ``bbox`` is the detector's box.
    """
    from pycocotools import mask

    mask_utils: Any = mask

    annotations: list[dict[str, Any]] = []
    for image_id, lab in enumerate(labels, start=1):
        for j, (row, label) in enumerate(zip(lab.rows, lab.labels, strict=True)):
            x1, y1, x2, y2, conf, cls = (float(v) for v in row)
            det_bbox = [x1, y1, x2 - x1, y2 - y1]
            ann: dict[str, Any] = {
                "image_id": image_id,
                "category_id": int(cls) + 1,
                "score": conf,
                "det_bbox": det_bbox,
                "iscrowd": 0,
                "attributes": {"label": label},
            }
            if lab.masks is None:
                ann |= {"bbox": det_bbox, "area": det_bbox[2] * det_bbox[3]}
            else:
                rle = lab.masks[j]
                area = float(mask_utils.area(rle))
                if not area:
                    continue
                ann |= {
                    "segmentation": {"size": rle["size"], "counts": rle["counts"].decode()},
                    "bbox": mask_utils.toBbox(rle).tolist(),
                    "area": area,
                }
            annotations.append(ann)
    return visdrone_vid.coco_file(images, annotations, visdrone_vid.COCO_CATEGORIES)


def label_frames(
    config: LabelingConfig, images: list[visdrone_vid.Image], device: str
) -> list[Labels]:
    labeler = Labeler(config, device)
    out = []
    for i, im in enumerate(images, start=1):
        out.append(labeler(read_rgb(im.path)))
        if i % 50 == 0:
            logger.info(f"  {i}/{len(images)} frames")
    del labeler
    gc.collect()
    return out


def run_trial(
    mlflow: Any,
    stage: str,
    name: str,
    config: LabelingConfig,
    images: list[visdrone_vid.Image],
    scoring_set: str,
    full: bool,
) -> None:
    from mlops_cv.config import get_settings

    settings = get_settings()
    data_yaml = visdrone_vid.scoring_dir(settings.data.autolabel_dir, scoring_set) / "data.yaml"
    labels = label_frames(config, images, settings.training.device)
    metrics = scores(images, labels, data_yaml, full=full) | latency(labels)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / autolabels_file(scoring_set)
        path.write_text(json.dumps(coco_autolabels(images, labels)))
        metrics["autolabels_mb"] = path.stat().st_size / 2**20
        with mlflow.start_run(run_name=f"{name}-{scoring_set}"):
            mlflow.set_tags({"autolabel.stage": stage, SET_TAG: scoring_set})
            mlflow.log_params(params(config) | {"frames": len(images)})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(path))
    logger.info(f"{name}: {json.dumps(metrics, indent=1)}")


def runnable(config: LabelingConfig, scoring_set: str) -> bool:
    """The champion trained on VID train, so it never labels a train scoring set."""
    return not (config.detector.model == CHAMPION and visdrone_vid.SETS[scoring_set][0] == "train")


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run one auto-labeling stage, one run per trial.")
    p.add_argument("stage", type=Path, help="configs/autolabel/<stage>.yaml")
    p.add_argument("--set", choices=visdrone_vid.SETS, default="train-sample", dest="scoring_set")
    p.add_argument("--limit", type=int, help="label and score only the first N frames")
    args = p.parse_args(argv)

    os.environ["MLFLOW__EXPERIMENT"] = AUTOLABEL_EXPERIMENT
    from mlops_cv.config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    trials = load_stage(args.stage)
    scoring_set = args.scoring_set
    images = visdrone_vid.prepare(settings.data.raw_dir, settings.data.autolabel_dir, scoring_set)
    full = args.limit is None
    images = images[: args.limit]

    from mlops_cv.tracking import client

    mlflow = client.connect(settings)
    failed = []
    for name, config in trials.items():
        if not runnable(config, scoring_set):
            logger.info(f"{name}: skipped, the champion trained on {scoring_set}'s split")
            continue
        logger.info(f"{name}: labeling {len(images)} {scoring_set} frames")
        try:
            run_trial(mlflow, args.stage.stem, name, config, images, scoring_set, full)
        except Exception:
            logger.exception(f"{name} failed")
            failed.append(name)
    if failed:
        logger.error(f"{len(failed)}/{len(trials)} trials failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
