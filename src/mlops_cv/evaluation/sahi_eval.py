"""Evaluation-only arms on a tuning trial's ``best.pt``: they measure, never select.

- ``o2m``: ``model.val(end2end=False)``, YOLO26's one-to-many head with NMS on the same weights,
  scored like the trial's own ``val/`` keys.
- ``sahi``: SAHI sliced inference (slices at the trial's imgsz, overlap 0.2, plus the full frame)
  against the same trial's full-frame predictions, both through ``score_detections`` at val's
  floor and cap, overall and by source resolution.

Each arm is one ``<trial>-<arm>`` run in the tune experiment. ``--slice N`` runs only the sahi
arm with N-px slices (each still predicted at imgsz) as ``<trial>-sahi<N>``.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mlops_cv.evaluation.zero_shot import CONF, MAX_DET
from mlops_cv.startup import exits_on_startup_error

logger = logging.getLogger(__name__)

OVERLAP = 0.2
NMS_IOU = 0.5
WARMUP = 3  # SAHI frames left out of the latency readings


def resolution_group(width: int) -> str:
    """The group a frame of this source width reports under: val holds 3840, 2688/2720 and
    1344/1904 frames. Test is mostly 1360 wide, so ``1904px`` is the group test can repeat."""
    return "3840px" if width >= 3840 else "2720px" if width >= 2688 else "1904px"


def top_k(rows: np.ndarray, k: int = MAX_DET) -> np.ndarray:
    """The ``k`` highest-confidence ``(n, 6)`` rows: ``model.val``'s cap; SAHI's merge has none."""
    return rows[np.argsort(-rows[:, 4], kind="stable")][:k]


def groups(sizes: Mapping[str, tuple[int, int]]) -> dict[str, list[str]]:
    """``{frame name: (w, h)}`` -> ``{resolution group: frame names}``."""
    out: dict[str, list[str]] = {}
    for name, (w, _) in sizes.items():
        out.setdefault(resolution_group(w), []).append(name)
    return dict(sorted(out.items()))


def _scored(
    detections: Mapping[str, np.ndarray],
    data_yaml: Path,
    prefix: str,
    frame_groups: Mapping[str, Collection[str]],
) -> dict[str, float]:
    """Headline, per-class and mAP50-95_small under ``prefix``; headline per resolution group."""
    from mlops_cv.evaluation.score_detections import score, score_with_ap_small
    from mlops_cv.tracking.metric_keys import (
        MAP_SMALL,
        PRIMARY,
        headline_metrics,
        metric_key,
        per_class_metrics,
    )
    from mlops_cv.training.train import TRIAL_SPLIT

    metrics, ap_s = score_with_ap_small(detections, data_yaml, TRIAL_SPLIT)
    out = (
        headline_metrics(metrics.box, prefix=prefix)
        | {metric_key(prefix, MAP_SMALL): ap_s}
        | per_class_metrics(
            metrics.maps, metrics.ap_class_index, metrics.names, prefix=metric_key(prefix, PRIMARY)
        )
    )
    for group, frames in frame_groups.items():
        box = score(detections, data_yaml, TRIAL_SPLIT, frames=frames).box
        out |= headline_metrics(box, prefix=metric_key(prefix, group))
    return out


def _latency(seconds: Iterable[float]) -> dict[str, float]:
    from mlops_cv.tracking.metric_keys import LATENCY_PREFIX, metric_key

    ms = np.asarray(list(seconds)) * 1000
    return {
        metric_key(LATENCY_PREFIX, "p50_ms"): float(np.percentile(ms, 50)),
        metric_key(LATENCY_PREFIX, "p95_ms"): float(np.percentile(ms, 95)),
        metric_key(LATENCY_PREFIX, "mean_ms"): float(ms.mean()),
    }


def sahi_arm(
    model: Any, images_dir: Path, data_yaml: Path, *, imgsz: int, slice_size: int, device: str
) -> dict:
    """SAHI under ``val/``, the path-matched full frame under ``val_full/``, latency, passes."""
    from PIL import Image
    from sahi.models.ultralytics import UltralyticsDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.slicing import get_slice_bboxes

    from mlops_cv.training.train import TRIAL_SPLIT

    images = sorted(images_dir.glob("*.jpg"))
    sizes = {p.name: Image.open(p).size for p in images}
    frame_groups = groups(sizes)

    full = {
        Path(r.path).name: r.boxes.data.cpu().numpy()
        for r in model.predict(
            str(images_dir),  # a directory streams frame by frame; a list is one batch
            imgsz=imgsz,
            device=device,
            conf=CONF,
            max_det=MAX_DET,
            stream=True,
            verbose=False,
        )
    }

    sliced = UltralyticsDetectionModel(
        model=model, confidence_threshold=CONF, image_size=imgsz, device=device
    )
    detections, seconds = {}, []
    for path in images:
        start = time.perf_counter()
        result = get_sliced_prediction(
            str(path),
            sliced,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=NMS_IOU,
            postprocess_class_agnostic=False,
            force_postprocess_type=True,
            verbose=0,
        )
        seconds.append(time.perf_counter() - start)
        rows = np.array(
            [
                [*p.bbox.to_xyxy(), p.score.value, p.category.id]
                for p in result.object_prediction_list
            ],
            np.float32,
        ).reshape(-1, 6)
        detections[path.name] = top_k(rows)

    passes = [
        len(get_slice_bboxes(h, w, slice_size, slice_size, False, OVERLAP, OVERLAP)) + 1
        for w, h in sizes.values()
    ]
    return (
        _scored(detections, data_yaml, TRIAL_SPLIT, frame_groups)
        | _scored(full, data_yaml, f"{TRIAL_SPLIT}_full", frame_groups)
        | _latency(seconds[WARMUP:])
        | {"sahi/passes_per_frame": float(np.mean(passes))}
    )


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    from mlops_cv.training.train import TRIAL_SPLIT
    from mlops_cv.training.tune import TUNE_EXPERIMENT

    p = argparse.ArgumentParser(description="Evaluation-only arms on tuning trials' best.pt.")
    p.add_argument("trials", nargs="+", help="trial run names, e.g. stage5-s1280-zoom0.75-1.5")
    p.add_argument("--slice", type=int, help="SAHI slice px (default: imgsz); sahi arm only")
    args = p.parse_args(argv)

    os.environ["MLFLOW__EXPERIMENT"] = TUNE_EXPERIMENT  # never production's experiment
    from mlops_cv.config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")

    from ultralytics import YOLO

    from mlops_cv.data.subset import IMAGES_DIRNAME, dataset_yaml
    from mlops_cv.evaluation.evaluate import split_metrics
    from mlops_cv.orchestration.handoff import best_weights_uri
    from mlops_cv.tracking import client
    from mlops_cv.tracking.resolve import resolve_model

    mlflow = client.connect(settings)
    data_yaml = dataset_yaml(settings)
    images_dir = settings.data.subset_dir / IMAGES_DIRNAME / TRIAL_SPLIT
    device = settings.training.device

    for trial in args.trials:
        runs = mlflow.search_runs(
            experiment_names=[TUNE_EXPERIMENT],
            filter_string=f"attributes.run_name = '{trial}' and attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
            output_format="list",
        )
        if not runs:
            raise ValueError(f"no finished trial run named {trial}")
        run = runs[0]
        imgsz = int(run.data.params["imgsz"])
        weights, _ = resolve_model(best_weights_uri(run.info.run_id))
        for arm in ("sahi",) if args.slice else ("o2m", "sahi"):
            model = YOLO(str(weights))
            if arm == "o2m":
                readings = split_metrics(
                    model,
                    data_yaml,
                    TRIAL_SPLIT,
                    imgsz=imgsz,
                    device=device,
                    batch=8,  # the trial's own val batch
                    name=f"{trial}-o2m",
                    end2end=False,
                )
            else:
                slice_size = args.slice or imgsz
                readings = sahi_arm(
                    model, images_dir, data_yaml, imgsz=imgsz, slice_size=slice_size, device=device
                )
            name = f"{trial}-{arm}{args.slice or ''}"
            with mlflow.start_run(run_name=name):
                mlflow.set_tag("tune.stage", "arms")
                mlflow.log_params({"trial": trial, "trial_run_id": run.info.run_id, "imgsz": imgsz})
                if arm == "sahi":
                    mlflow.log_param("slice", slice_size)
                mlflow.log_metrics(readings)
            logger.info(f"{name}: {readings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
