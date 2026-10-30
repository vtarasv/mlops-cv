"""Run one labeling stage: ``configs/autolabel/<stage>.yaml`` -> one MLflow run per labeling trial.

Each trial labels a whole VID scoring set. The filtered detector boxes (``det``) and the
mask-derived boxes (``seg``) are scored apart, through ultralytics' validator, after the
ignored-region drop; a sequence-mode trial also scores its tracks through TrackEval. The
auto-labels are logged as one COCO RLE file per trial and set.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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

from mlops_cv.autolabel import sequence, track_scores, visdrone_vid
from mlops_cv.autolabel.labeler import (
    BOX_TRACKERS,
    CHAMPION,
    OWN,
    Labeler,
    LabelingConfig,
    Labels,
    Tracker,
    mask_areas,
    read_rgb,
)
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
    keep = mask_areas(labels.masks) > 0
    rows = np.column_stack([xywh[:, :2], xywh[:, :2] + xywh[:, 2:], labels.rows[:, 4:6]])
    return rows[keep]


def track_rows(labels: Labels) -> np.ndarray:
    """``(n, 7)``: the auto-labels' boxes (mask boxes when masked), conf, class and track id."""
    ids = np.asarray(labels.track_ids)
    if labels.masks is None:
        return np.column_stack([labels.rows, ids])
    keep = mask_areas(labels.masks) > 0
    return np.column_stack([mask_rows(labels), ids[keep]])


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
    images: list[visdrone_vid.Image],
    labels: list[Labels],
    data_yaml: Path,
    *,
    full: bool,
    keys: list[Labels | None] | None = None,
) -> dict[str, float]:
    """Each box kind scored on every image, then on the ``SUBSET_CAP`` subset.

    ``keys`` (sequence mode) holds each frame's key-frame detections, ``None`` between key frames:
    ``det`` then scores the key frames only, and ``seg`` every frame's auto-labels.
    """
    from mlops_cv.tracking.metric_keys import metric_key

    keyed = [(im, k) for im, k in zip(images, keys or labels, strict=True) if k is not None]
    kinds = {DET: ([im for im, _ in keyed], [k.rows for _, k in keyed])}
    if labels and labels[0].masks is not None:
        kinds[SEG] = (images, [mask_rows(lab) for lab in labels])
    out: dict[str, float] = {}
    for kind, (ims, rows) in kinds.items():
        frames = None if full and len(ims) == len(images) else {im.name for im in ims}
        detections = {im.name: outside_ignored(r, im) for im, r in zip(ims, rows, strict=True)}
        subset = set(uncapped(ims, SUBSET_CAP))
        out |= readings(detections, data_yaml, kind, frames)
        out |= readings(detections, data_yaml, f"{kind}/uncapped{SUBSET_CAP}", subset)
        out[metric_key(kind, "boxes_per_image")] = float(np.mean([len(r) for r in rows]))
    out[metric_key(DET, "detected_per_image")] = float(np.mean([k.detected for _, k in keyed]))
    cap = f"gt/cap{SUBSET_CAP}"
    out[metric_key(cap, "recall_ceiling")] = recall_ceiling(images, SUBSET_CAP)
    out[metric_key(cap, "uncapped_images")] = float(len(uncapped(images, SUBSET_CAP)))
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


def amortised_latency(labels: list[Labels], tracker: Tracker) -> dict[str, float]:
    """Each stage's total wall-clock over the frames, divided by the frame count.

    The warm-up drops whole key-frame cycles, so the key frames keep their share.
    """
    from mlops_cv.tracking.metric_keys import LATENCY_PREFIX, metric_key

    interval = 1 if tracker.model in BOX_TRACKERS else tracker.key_interval
    timed = labels[-(-WARMUP // interval) * interval :] or labels
    stages = {s for lab in timed for s in lab.ms}
    ms = {s: sum(lab.ms.get(s, 0.0) for lab in timed) / len(timed) for s in stages}
    ms["total"] = sum(ms.values())
    return {metric_key(metric_key(LATENCY_PREFIX, s), "amortised_ms"): v for s, v in ms.items()}


def coco_autolabels(
    images: list[visdrone_vid.Image], labels: list[Labels], floor: float | None = None
) -> dict[str, Any]:
    """The trial's COCO file: ``bbox`` is the mask's box, the detector's goes in ``det_bbox``.

    An empty mask, or a score below ``floor``, is no auto-label. Without a segmenter, ``bbox`` is
    the detector's box. In sequence mode each auto-label has a ``track_id``, and a tracked frame
    has no ``det_bbox``.
    """
    from pycocotools import mask

    mask_utils: Any = mask

    annotations: list[dict[str, Any]] = []
    for image_id, lab in enumerate(labels, start=1):
        for j, (row, label) in enumerate(zip(lab.rows, lab.labels, strict=True)):
            x1, y1, x2, y2, conf, cls = (float(v) for v in row)
            if floor is not None and conf < floor:
                continue
            det_bbox = [x1, y1, x2 - x1, y2 - y1]
            ann: dict[str, Any] = {
                "image_id": image_id,
                "category_id": int(cls) + 1,
                "score": conf,
                "iscrowd": 0,
                "attributes": {"label": label},
            }
            if np.isfinite(det_bbox).all():
                ann["det_bbox"] = det_bbox
            if lab.track_ids is not None:
                ann["track_id"] = int(lab.track_ids[j])
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


def detections_store(scoring_dir: Path, config: LabelingConfig) -> Path:
    """Where a whole scoring set's detector boxes, before the merge and box filter, are stored."""
    # A new field keeps old keys; the merge runs on stored tiled passes, so it isn't keyed.
    detector = config.detector.model_dump_json(exclude_none=True, exclude={"tiling": {"merge"}})
    key = hashlib.sha256(detector.encode()).hexdigest()[:16]
    return scoring_dir / "detections" / f"{key}.npz"


def save_detections(path: Path, images: list[visdrone_vid.Image], raw: list[Labels]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "names": [im.name for im in images],
        "counts": [len(r.rows) for r in raw],
        "rows": np.concatenate([r.rows for r in raw]).reshape(-1, 6),
        "labels": np.array([label for r in raw for label in r.labels], str),
        "detector_ms": [r.ms["detector"] for r in raw],
    }
    if windows := [r.windows for r in raw if r.windows is not None]:
        arrays["windows"] = np.concatenate(windows)
    np.savez_compressed(path, **arrays)


def load_detections(path: Path, images: list[visdrone_vid.Image]) -> list[Labels]:
    """The stored boxes, with the detector latency they were measured at."""
    z = np.load(path)
    if z["names"].tolist() != [im.name for im in images]:
        raise ValueError(f"{path} holds other frames: delete it to re-detect")
    split = np.cumsum(z["counts"])[:-1]
    rows, labels = np.split(z["rows"], split), np.split(z["labels"], split)
    windows = np.split(z["windows"], split) if "windows" in z else [None] * len(rows)
    parts = zip(rows, labels, z["detector_ms"], windows, strict=True)
    return [
        Labels(r, lb.tolist(), None, {"detector": float(ms)}, len(r), windows=win)
        for r, lb, ms, win in parts
    ]


def label_frames(
    config: LabelingConfig,
    images: list[visdrone_vid.Image],
    device: str,
    store: Path | None = None,
) -> list[Labels]:
    """Each frame's auto-labels; with ``store``, the detector's boxes are read from it or written.

    A stored detector pass reruns only the box filter and segmenter, so filter settings rescore
    without the detector. The store is keyed by the detector's settings, not its code: delete
    it after changing the detector code.
    """
    raw = load_detections(store, images) if store and store.exists() else None
    if raw:
        logger.info(f"  detector boxes from {store}")
    labeler = Labeler(config, device, detect=raw is None)
    out, detected = [], []
    for i, im in enumerate(images, start=1):
        rgb = read_rgb(im.path)
        det = raw[i - 1] if raw else labeler.detections(rgb)
        if store and raw is None:
            detected.append(det)
        out.append(labeler.finish(rgb, det))
        if i % 50 == 0:
            logger.info(f"  {i}/{len(images)} frames")
    if store and raw is None:
        save_detections(store, images, detected)
    del labeler
    gc.collect()
    return out


def label_sequences(
    config: LabelingConfig,
    images: list[visdrone_vid.Image],
    device: str,
    detections: dict[str, list[Labels]],
    store: Path | None,
) -> tuple[list[Labels], list[Labels | None], list[int]]:
    """Per frame: the auto-labels, the key-frame detections and the live tracks.

    Box trackers share one detection pass per detector, filter and segmenter, cached in
    ``detections``.
    """
    assert config.tracker is not None
    if config.tracker.model in BOX_TRACKERS:
        key = config.model_copy(update={"tracker": None}).model_dump_json()
        if key not in detections:
            detections[key] = label_frames(config, images, device, store)
        tracked = list(sequence.box_tracks(config.tracker.model, images, detections[key]))
        return [lab for lab, _ in tracked], list(detections[key]), [n for _, n in tracked]
    labels, keys, live = zip(*sequence.video_tracks(config, images, device), strict=True)
    return list(labels), list(keys), list(live)


def run_trial(
    mlflow: Any,
    stage: str,
    name: str,
    config: LabelingConfig,
    images: list[visdrone_vid.Image],
    scoring_set: str,
    full: bool,
    detections: dict[str, list[Labels]],
) -> None:
    from mlops_cv.config import get_settings

    settings = get_settings()
    scoring_dir = visdrone_vid.scoring_dir(settings.data.autolabel_dir, scoring_set)
    data_yaml = scoring_dir / "data.yaml"
    # the stored boxes carry no masks, so own masks rerun the detector
    store = detections_store(scoring_dir, config) if full and config.segmenter != OWN else None
    device = settings.training.device
    if config.tracker is None:
        labels = label_frames(config, images, device, store)
        metrics = scores(images, labels, data_yaml, full=full) | latency(labels)
    else:
        labels, keys, live = label_sequences(config, images, device, detections, store)
        pairs = zip(images, labels, strict=True)
        tracked = [outside_ignored(track_rows(lab), im) for im, lab in pairs]
        metrics = (
            scores(images, labels, data_yaml, full=full, keys=keys)
            | amortised_latency(labels, config.tracker)
            | track_scores.readings(images, tracked)
            | {"tracks/peak_live": float(max(live))}
        )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / autolabels_file(scoring_set)
        path.write_text(json.dumps(coco_autolabels(images, labels, config.confidence_floor)))
        metrics["autolabels_mb"] = path.stat().st_size / 2**20
        with mlflow.start_run(run_name=f"{name}-{scoring_set}"):
            mlflow.set_tags({"autolabel.stage": stage, SET_TAG: scoring_set})
            mlflow.log_params(params(config) | {"frames": len(images)})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(path))
    logger.info(f"{name}: {json.dumps(metrics, indent=1)}")


def skip_reason(config: LabelingConfig, scoring_set: str) -> str | None:
    """Why a trial can't run on a scoring set, or ``None``."""
    if config.detector.model == CHAMPION and visdrone_vid.SETS[scoring_set][0] == "train":
        return f"the champion trained on {scoring_set}'s split"
    if config.tracker and scoring_set not in visdrone_vid.SEQUENCE_SETS:
        return f"tracking needs whole sequences: {', '.join(visdrone_vid.SEQUENCE_SETS)}"
    return None


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run one auto-labeling stage, one run per trial.")
    p.add_argument("stage", type=Path, help="configs/autolabel/<stage>.yaml")
    p.add_argument("--set", choices=visdrone_vid.SETS, default="train-sample", dest="scoring_set")
    p.add_argument("--limit", type=int, help="label and score only the first N frames")
    p.add_argument("--sequence", action="append", dest="sequences", help="only these sequences")
    args = p.parse_args(argv)

    os.environ["MLFLOW__EXPERIMENT"] = AUTOLABEL_EXPERIMENT
    from mlops_cv.config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    trials = load_stage(args.stage)
    scoring_set = args.scoring_set
    images = visdrone_vid.prepare(settings.data.raw_dir, settings.data.autolabel_dir, scoring_set)
    if args.sequences:
        images = [im for im in images if im.sequence in args.sequences]
    full = args.limit is None and not args.sequences
    images = images[: args.limit]

    from mlops_cv.tracking import client

    mlflow = client.connect(settings)
    failed = []
    detections: dict[str, list[Labels]] = {}
    for name, config in trials.items():
        if reason := skip_reason(config, scoring_set):
            logger.info(f"{name}: skipped, {reason}")
            continue
        logger.info(f"{name}: labeling {len(images)} {scoring_set} frames")
        try:
            run_trial(mlflow, args.stage.stem, name, config, images, scoring_set, full, detections)
        except Exception:
            logger.exception(f"{name} failed")
            failed.append(name)
    if failed:
        logger.error(f"{len(failed)}/{len(trials)} trials failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
