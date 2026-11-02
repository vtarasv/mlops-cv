"""Run one labeling stage: ``configs/tuning/<case>/<stage>.yaml`` -> one MLflow run per trial.

A trial without a tracker labels a subset split's frames and is scored by ultralytics' validator
against the split's VisDrone-rule copy, its boxes through the same rule. A trial with a tracker
labels the split's full raw clips and scores its tracks through TrackEval. Each logs its boxes
as VisDrone-VID annotation files, one per sequence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from mlops_cv.autolabel.labeler import Labels, annotation_lines, read_rgb
from mlops_cv.config.settings import BoxFilter
from mlops_cv.data.subset import IMAGES_DIRNAME
from mlops_cv.startup import StartupError, exits_on_startup_error
from mlops_cv.tuning import common
from mlops_cv.tuning.autolabel_visdrone_vid import track_scores, tracks, visdrone_vid
from mlops_cv.tuning.autolabel_visdrone_vid.open_vocab import Detector, TrialLabeler
from mlops_cv.tuning.autolabel_visdrone_vid.tracks import BOX_TRACKERS, Tracker

logger = logging.getLogger(__name__)

AUTOLABEL_EXPERIMENT = "aerial-object-detection-autolabel-tune"
# Stored detector passes and downloaded trials; the FiftyOne App mounts it at the same path.
RUNS_DIR = Path("runs/autolabel")
ANNOTATIONS = "annotations"  # the trial's artifact dir: one VisDrone-VID file per sequence
STAGE_TAG, SPLIT_TAG = "autolabel.stage", "autolabel.split"
# SAM 3's per-prompt cap. Every row is also scored on the images where no raw category has
# more ground truth than this, so a capped row compares on the images its cap can't bind.
SUBSET_CAP = 200
# Images left out of the latency readings.
WARMUP = 3


class LabelingConfig(BaseModel):
    """One labeling trial's settings; unset ones are production's (``AutolabelSettings``)."""

    model_config = ConfigDict(extra="forbid")

    detector: Detector = Detector()
    box_filter: BoxFilter = BoxFilter()
    tracker: Tracker | None = None  # ``None`` labels each frame on its own
    # The track thresholds and P/R readings; scoring and stored boxes aren't cut at it.
    confidence_floor: float | None = None


def load_stage(path: Path) -> dict[str, LabelingConfig]:
    """``{run name: labeling config}``, through the tuning grid's own stage loader."""
    return {name: LabelingConfig.model_validate(k) for name, k in common.load_stage(path).items()}


def params(config: LabelingConfig) -> dict[str, Any]:
    """The config flattened to ``section.key`` MLflow params."""
    return {
        f"{key}.{k}" if isinstance(value, dict) else key: v
        for key, value in config.model_dump().items()
        for k, v in (value.items() if isinstance(value, dict) else [(key, value)])
    }


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


def scores(
    images: list[visdrone_vid.Image],
    labels: list[Labels],
    data_yaml: Path,
    prefix: str,
    frames: Collection[str] | None,
    floor: float | None = None,
) -> dict[str, float]:
    """The validator's readings under ``prefix``, on every image and on the ``SUBSET_CAP`` ones."""
    from mlops_cv.tracking.metric_keys import metric_key

    rows = {im.name: lab.rows for im, lab in zip(images, labels, strict=True)}
    detections = visdrone_vid.drop_all_ignored(rows, images)
    subset = set(uncapped(images, SUBSET_CAP))
    cap = f"gt/cap{SUBSET_CAP}"
    return (
        visdrone_vid.readings(detections, data_yaml, "val", prefix, frames, floor)
        | visdrone_vid.readings(
            detections, data_yaml, "val", f"{prefix}/uncapped{SUBSET_CAP}", subset, floor
        )
        | {
            metric_key(prefix, "boxes_per_image"): float(np.mean([len(r) for r in rows.values()])),
            metric_key(prefix, "detected_per_image"): float(np.mean([x.detected for x in labels])),
            metric_key(cap, "recall_ceiling"): recall_ceiling(images, SUBSET_CAP),
            metric_key(cap, "uncapped_images"): float(len(subset)),
        }
    )


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


def write_annotations(out: Path, images: list[visdrone_vid.Image], labels: list[Labels]) -> None:
    """The trial's boxes as one VisDrone-VID annotation file per sequence under ``out``."""
    lines: dict[str, list[str]] = defaultdict(list)
    for im, lab in zip(images, labels, strict=True):
        lines[im.sequence] += annotation_lines(im.frame, lab)
    for sequence, text in lines.items():
        (out / f"{sequence}.txt").write_text("".join(f"{line}\n" for line in text))


def detections_store(frames: str, config: LabelingConfig) -> Path:
    """Where a whole frame set's detector boxes, before the merge and box filter, are stored."""
    # A new field keeps old keys; the merge runs on stored tiled passes, so it isn't keyed.
    detector = config.detector.model_dump_json(exclude_none=True, exclude={"tiling": {"merge"}})
    key = hashlib.sha256(detector.encode()).hexdigest()[:16]
    return RUNS_DIR / "detections" / f"{frames}-{key}.npz"


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
    """``images``' stored boxes, with the detector latency they were measured at."""
    z = np.load(path)
    split = np.cumsum(z["counts"])[:-1]
    rows, labels = np.split(z["rows"], split), np.split(z["labels"], split)
    windows = np.split(z["windows"], split) if "windows" in z else [None] * len(rows)
    parts = zip(z["names"].tolist(), rows, labels, z["detector_ms"], windows, strict=True)
    stored = {
        name: Labels(r, lb.tolist(), {"detector": float(ms)}, len(r), windows=win)
        for name, r, lb, ms, win in parts
    }
    if missing := [im.name for im in images if im.name not in stored]:
        raise ValueError(f"{path} lacks {missing[0]} and {len(missing) - 1} more: delete it")
    return [stored[im.name] for im in images]


def label_frames(
    config: LabelingConfig,
    images: list[visdrone_vid.Image],
    device: str,
    store: Path | None = None,
) -> list[Labels]:
    """Each frame's boxes; with ``store``, the detector's boxes are read from it or written.

    A stored detector pass reruns only the box filter, so filter settings rescore without the
    detector. The store is keyed by the detector's settings, not its code: delete it after
    changing the detector code.
    """
    raw = load_detections(store, images) if store and store.exists() else None
    if raw:
        logger.info(f"  detector boxes from {store}")
    labeler = TrialLabeler(
        config.detector, config.box_filter, device, common.MAX_DET, detect=raw is None
    )
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
) -> tuple[list[Labels], list[int]]:
    """Per frame: the tracked auto-labels and the live tracks.

    Box trackers share one detection pass per detector and filter, cached in ``detections``.
    Video trackers read their key frames' boxes from ``store`` when it exists.
    """
    assert config.tracker is not None
    floor = config.confidence_floor
    if config.tracker.model in BOX_TRACKERS:
        # the floor only sets the track thresholds, so floors share the pass
        shared = {"tracker": None, "confidence_floor": None}
        key = config.model_copy(update=shared).model_dump_json()
        if key not in detections:
            detections[key] = label_frames(config, images, device, store)
        tracked = list(tracks.box_tracks(config.tracker.model, images, detections[key], floor))
        return [lab for lab, _ in tracked], [n for _, n in tracked]
    raw = load_detections(store, images) if store and store.exists() else None
    labeler = TrialLabeler(
        config.detector, config.box_filter, device, common.MAX_DET, detect=raw is None
    )
    out = list(tracks.video_tracks(labeler, config.tracker, images, raw, floor))
    del labeler
    gc.collect()
    return [lab for lab, _, _ in out], [n for _, _, n in out]


def run_trial(
    mlflow: Any,
    stage: str,
    name: str,
    config: LabelingConfig,
    images: list[visdrone_vid.Image],
    split: str,
    full: bool,
    detections: dict[str, list[Labels]],
) -> None:
    from mlops_cv.config import get_settings

    settings = get_settings()
    frames = f"{split}-clips" if config.tracker else split
    store = detections_store(frames, config)
    store = store if full or store.exists() else None
    device = settings.training.device
    if config.tracker is None:
        labels = label_frames(config, images, device, store)
        subset = None if full else {im.name for im in images}
        data_yaml = visdrone_vid.rule_yaml(settings, split)
        floor = config.confidence_floor
        metrics = scores(images, labels, data_yaml, split, subset, floor) | latency(labels)
    else:
        labels, live = label_sequences(config, images, device, detections, store)
        pairs = zip(images, labels, strict=True)
        # (n, 7) rows: x1 y1 x2 y2 conf class track-id
        rows = [(im, np.column_stack([lab.rows, np.asarray(lab.track_ids)])) for im, lab in pairs]
        tracked = [visdrone_vid.drop_ignored(r, im) for im, r in rows]
        metrics = (
            amortised_latency(labels, config.tracker)
            | track_scores.readings(images, tracked)
            | {"tracks/peak_live": float(max(live))}
        )
    with tempfile.TemporaryDirectory() as tmp:
        write_annotations(Path(tmp), images, labels)
        with mlflow.start_run(run_name=f"{name}-{split}"):
            mlflow.set_tags({STAGE_TAG: stage, SPLIT_TAG: split})
            mlflow.log_params(params(config) | {"frames": len(images)})
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(tmp, ANNOTATIONS)
    logger.info(f"{name}: {json.dumps(metrics, indent=1)}")


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run one labeling stage, one run per labeling trial.")
    p.add_argument("stage", type=Path, help="configs/tuning/autolabel-visdrone-vid/<stage>.yaml")
    p.add_argument("--split", choices=visdrone_vid.RULE_SPLITS, default="train")
    p.add_argument("--limit", type=int, help="label and score only the first N frames")
    p.add_argument("--sequence", action="append", dest="sequences", help="only these sequences")
    args = p.parse_args(argv)

    os.environ["MLFLOW__EXPERIMENT"] = AUTOLABEL_EXPERIMENT
    from mlops_cv.config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    trials = load_stage(args.stage)
    rule = settings.data.subset_dir / IMAGES_DIRNAME / visdrone_vid.visdrone_rule_split(args.split)
    if not rule.is_dir():
        raise StartupError(
            f"no {rule}: run python -m mlops_cv.tuning.autolabel_visdrone_vid.visdrone_vid"
        )
    full = args.limit is None and not args.sequences
    frame_sets: dict[bool, list[visdrone_vid.Image]] = {}

    def frames(clips: bool) -> list[visdrone_vid.Image]:
        if clips not in frame_sets:
            data = settings.data
            images = visdrone_vid.read_split(data.raw_dir, data.subset_dir, args.split, clips)
            if args.sequences:
                images = [im for im in images if im.sequence in args.sequences]
            frame_sets[clips] = images[: args.limit]
        return frame_sets[clips]

    from mlops_cv.tracking import client

    mlflow = client.connect(settings)
    failed = []
    detections: dict[str, list[Labels]] = {}
    for name, config in trials.items():
        images = frames(config.tracker is not None)
        logger.info(f"{name}: labeling {len(images)} {args.split} frames")
        try:
            run_trial(mlflow, args.stage.stem, name, config, images, args.split, full, detections)
        except Exception:
            logger.exception(f"{name} failed")
            failed.append(name)
    if failed:
        logger.error(f"{len(failed)}/{len(trials)} trials failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
