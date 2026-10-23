"""The third-party baseline: dronefreak's DET-trained YOLO26 s/m @1280 on test, beside ours.

A different protocol, never a comparability claim. The checkpoints predict VisDrone's 10 raw
categories (+ an empty ``others``), so they are scored like the zero-shot baselines: predict,
remap by name onto the merged classes, same-box dedupe, ``score_detections``. Never ``model.val``:
it scores the model's own class ids against the merged labels. ``stage2-s1280`` and
``stage2-m1280`` go through the same path, so both sides are read on the same frames.

Each model x head is one run in the tune experiment tagged ``tune.stage=test-read``: ``test/``
keys over all 17 test sequences, ``test-no-det-overlap/`` keys over the 13 with no known
VisDrone-DET overlap (the headline). The checkpoints are AGPL-3.0: evaluated, never logged or
redistributed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path

from mlops_cv.startup import exits_on_startup_error

logger = logging.getLogger(__name__)

SPLIT = "test"
NO_OVERLAP = "test-no-det-overlap"

# run name -> (Hugging Face repo, pinned revision, best.pt sha256)
CHECKPOINTS: dict[str, tuple[str, str, str]] = {
    "dronefreak-yolo26s1280": (
        "dronefreak/visdrone-yolo26s",
        "dc042f4b188e76413f256b5ff69653f11a38a854",
        "98a66ac96e4470ab7cf118bf8c700fa65f7078a05f32578fca43a4c398c01879",
    ),
    "dronefreak-yolo26m1280": (
        "dronefreak/visdrone-yolo26m",
        "cf0d3e8ecde426e45f682331f7a1cc0c06f036ff",
        "eceacffc05defea59fde06691d3dc947a2aa67a07cd0ba4ee4390277136fed79",
    ),
}
LICENCE = "AGPL-3.0"
FINALISTS = ("stage2-s1280", "stage2-m1280")
IMGSZ = 1280

# Test sequences with frames in VisDrone-DET train or val (measured in research 02).
DET_OVERLAP = frozenset(
    {"uav0000201_00000_v", "uav0000297_00000_v", "uav0000249_00001_v", "uav0000249_02688_v"}
)

# Same merge as ingest's ``CLASS_MERGE``, keyed by the raw category names a DET model carries.
VISDRONE_TO_MERGED: dict[str, str] = {
    "pedestrian": "person",
    "people": "person",
    "car": "vehicle",
    "van": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "bicycle": "two-three-wheeler",
    "tricycle": "two-three-wheeler",
    "awning-tricycle": "two-three-wheeler",
    "motor": "two-three-wheeler",
}


def _download(repo: str, revision: str, sha256: str, dest: Path) -> Path:
    """``best.pt`` at a pinned revision, checked against its hash; kept and reused at ``dest``."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{repo}/resolve/{revision}/best.pt"
        urllib.request.urlretrieve(url, dest)  # noqa: S310 (a fixed https URL)
    if (got := hashlib.sha256(dest.read_bytes()).hexdigest()) != sha256:
        raise ValueError(f"{dest}: sha256 {got}, expected {sha256} ({repo}@{revision})")
    return dest


@exits_on_startup_error
def main() -> int:
    from mlops_cv.training.tune import TUNE_EXPERIMENT, finished_trial

    os.environ["MLFLOW__EXPERIMENT"] = TUNE_EXPERIMENT  # never production's experiment
    from mlops_cv.config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")

    from ultralytics import YOLO

    from mlops_cv.data.convert_visdrone_vid import YOLO_NAMES
    from mlops_cv.data.subset import IMAGES_DIRNAME, dataset_yaml
    from mlops_cv.evaluation.evaluate import latency_stub
    from mlops_cv.evaluation.score_detections import score_with_ap_small
    from mlops_cv.evaluation.zero_shot import CONF, MAX_DET, _rows, class_lut, merge_classes
    from mlops_cv.orchestration.handoff import best_weights_uri
    from mlops_cv.tracking import client
    from mlops_cv.tracking.metric_keys import (
        MAP_SMALL,
        PRIMARY,
        headline_metrics,
        metric_key,
        per_class_metrics,
    )
    from mlops_cv.tracking.resolve import resolve_model

    mlflow = client.connect(settings)
    data_yaml = dataset_yaml(settings)
    images_dir = settings.data.subset_dir / IMAGES_DIRNAME / SPLIT
    device = settings.training.device
    no_overlap = {
        p.name for p in images_dir.glob("*.jpg") if p.name.rsplit("_", 1)[0] not in DET_OVERLAP
    }

    # run name -> (weights, class table, params)
    models: dict[str, tuple[Path, dict[str, str], dict[str, str]]] = {}
    for name, (repo, revision, sha256) in CHECKPOINTS.items():
        dest = Path(settings.training.weights).parent / "third-party" / f"{name}.pt"
        params = {"source": f"hf://{repo}@{revision}", "sha256": sha256, "licence": LICENCE}
        models[name] = (_download(repo, revision, sha256, dest), VISDRONE_TO_MERGED, params)
    for trial in FINALISTS:
        run = finished_trial(mlflow, trial)
        weights, _ = resolve_model(best_weights_uri(run.info.run_id))
        params = {"trial": trial, "trial_run_id": run.info.run_id}
        models[f"{trial}-predict"] = (weights, {n: n for n in YOLO_NAMES.values()}, params)

    for name, (weights, table, params) in models.items():
        for end2end in (True, False):
            model = YOLO(str(weights))  # the predictor fixes the head on first use: one per head
            lut = class_lut(model.names, table)
            detections = {
                Path(r.path).name: merge_classes(_rows(r), lut)
                for r in model.predict(
                    str(images_dir),
                    imgsz=IMGSZ,
                    device=device,
                    conf=CONF,
                    max_det=MAX_DET,
                    end2end=end2end,
                    stream=True,
                    verbose=False,
                )
            }
            readings: dict[str, float] = {}
            for prefix, frames in ((SPLIT, None), (NO_OVERLAP, no_overlap)):
                metrics, ap_s = score_with_ap_small(detections, data_yaml, SPLIT, frames)
                readings |= (
                    headline_metrics(metrics.box, prefix=prefix)
                    | {metric_key(prefix, MAP_SMALL): ap_s}
                    | per_class_metrics(
                        metrics.maps,
                        metrics.ap_class_index,
                        metrics.names,
                        prefix=metric_key(prefix, PRIMARY),
                    )
                )
            if name in CHECKPOINTS:  # a trial's own latency readings stand
                readings |= latency_stub(model, images_dir, imgsz=IMGSZ, device=device)
            run_name = f"{name}-{SPLIT}" + ("" if end2end else "-o2m")
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("tune.stage", "test-read")
                mlflow.log_params(
                    params | {"imgsz": IMGSZ, "conf": CONF, "max_det": MAX_DET, "end2end": end2end}
                )
                mlflow.log_metrics(readings)
            logger.info(f"{run_name}: {readings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
