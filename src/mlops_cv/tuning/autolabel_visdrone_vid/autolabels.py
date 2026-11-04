"""The auto-labeled train split: production's auto-labels on the subset's train frames.

The stored train-clip pass is tracked as ``label_clip`` tracks it (``label_eval.label_sequences``,
pinned to it by a unit test), so the detector doesn't rerun. Each subset frame keeps the
auto-labels at the floor and outside its ignored regions (VisDrone's rule), as YOLO lines from
ingestion's converter. YOLO trains on it: ``configs/tuning/yolo26-visdrone-vid/autolabels.yaml``.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from mlops_cv.autolabel.labeler import Labels, above, annotation_lines
from mlops_cv.data.convert_visdrone_vid import YOLO_NAMES, convert_sequence_text
from mlops_cv.data.subset import IMAGES_DIRNAME, label_relpath
from mlops_cv.startup import exits_on_startup_error
from mlops_cv.tuning.autolabel_visdrone_vid import label_eval, visdrone_vid

logger = logging.getLogger(__name__)

SPLIT = "train"
# The labeling config of ``AutolabelSettings``' defaults (a unit test pins the two).
LABELER_STAGE = Path("configs/tuning/autolabel-visdrone-vid/labeler.yaml")


def yolo_labels(image: visdrone_vid.Image, labels: Labels, floor: float) -> list[str]:
    """One frame's auto-labels as ``label_clip`` writes them, through the rule, as YOLO lines."""
    lines = annotation_lines(image.frame, above(labels, floor))
    xywh = np.array([line.split(",")[2:6] for line in lines], float).reshape(-1, 4)
    keep = visdrone_vid.outside_ignored(xywh, image.ignored, image.width, image.height)
    text = "\n".join(line for line, kept in zip(lines, keep, strict=True) if kept)
    boxes = convert_sequence_text(text, (image.width, image.height)).get(image.frame, [])
    return [box.to_line() for box in boxes]


def class_counts(lines: Iterable[str]) -> Counter[str]:
    return Counter(YOLO_NAMES[int(line.split()[0])] for line in lines)


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    from mlops_cv.config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    data = settings.data
    (config,) = label_eval.load_stage(LABELER_STAGE).values()
    assert config.confidence_floor is not None
    clips = visdrone_vid.read_split(data.raw_dir, data.subset_dir, SPLIT, clips=True)
    store = label_eval.detections_store(f"{SPLIT}-clips", config)
    tracked, _ = label_eval.label_sequences(config, clips, settings.training.device, {}, store)
    wanted = visdrone_vid.subset_frames(data.subset_dir, SPLIT)
    frames = [
        (im, lab)
        for im, lab in zip(clips, tracked, strict=True)
        if im.frame in wanted.get(im.sequence, ())
    ]
    labels = {im.name: yolo_labels(im, lab, config.confidence_floor) for im, lab in frames}
    visdrone_vid.write_split(data.subset_dir, visdrone_vid.AUTOLABEL_SPLIT, SPLIT, labels)

    floored = sum(len(above(lab, config.confidence_floor).rows) for _, lab in frames)
    auto = class_counts(line for lines in labels.values() for line in lines)
    truth = class_counts(
        line
        for name in labels
        for line in (data.subset_dir / label_relpath(SPLIT, Path(name).stem))
        .read_text()
        .splitlines()
    )
    out = data.subset_dir / IMAGES_DIRNAME / visdrone_vid.AUTOLABEL_SPLIT
    dropped = floored - auto.total()
    logger.info(f"{len(labels)} frames, {dropped} auto-labels at the floor dropped by the rule")
    for name in [*YOLO_NAMES.values(), "total"]:
        a, t = (auto.total(), truth.total()) if name == "total" else (auto[name], truth[name])
        logger.info(f"{name}: {a} auto-labels, {t} ground truth ({a / t:.2f}x)")
    logger.info(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
