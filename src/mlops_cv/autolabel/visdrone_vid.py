"""Raw VisDrone-VID frames: their annotations, the toolkit's ignored-region rule and scoring sets.

A scoring set is a fixed choice of frames from one VID split. It is written to
``<autolabel_dir>/<set>/``: ``images`` (one link per frame, named ``<sequence>_<frame>.jpg``),
3-class YOLO ``labels``, ``data.yaml``, and COCO ``ground_truth.json`` + ``ignored.json`` for the
annotation editor. Ground truth is dropped by the ignored-region rule before it is written.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image as PILImage

from mlops_cv.data.convert_visdrone_vid import CATEGORY_COL, CLASS_MERGE, FRAME_COL, YOLO_NAMES
from mlops_cv.data.subset import frame_name, raw_split_dir
from mlops_cv.startup import exits_on_startup_error

logger = logging.getLogger(__name__)

RAW_NAMES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}
IGNORED_CATEGORY = 0
BOX_COLS = slice(2, 6)  # left, top, width, height
TRACK_COL = 1  # target_id
# The toolkit drops a box when at least this share of its w*h lies inside the ignored regions.
IGNORED_SHARE = 0.5
# The split every scoring dataset YAML names: ultralytics' check needs ``val:``.
SCORING_SPLIT = "val"
# Frames per sequence in the image-mode sample.
SAMPLE_FRAMES = 10


def evenly_spaced(n: int) -> Callable[[list[int]], list[int]]:
    """``n`` frames spread evenly over a sequence, first and last included."""
    return lambda frames: [
        frames[i] for i in np.unique(np.linspace(0, len(frames) - 1, n).round().astype(int))
    ]


def strided(stride: int) -> Callable[[list[int]], list[int]]:
    """Every ``stride``-th 1-based frame, as the repo's VID subset picks them."""
    return lambda frames: [f for f in frames if (f - 1) % stride == 0]


# Scoring set -> (YOLO split, frame picker per sequence).
SETS: dict[str, tuple[str, Callable[[list[int]], list[int]]]] = {
    "train-sample": ("train", evenly_spaced(SAMPLE_FRAMES)),  # image mode tunes here
    "val-stride20": ("val", strided(20)),  # the image-mode finalists meet here
    "val-full": ("val", strided(1)),  # sequence mode tunes here
    "test-dev-full": ("test", strided(1)),  # the finalists are reported here once
}
# The full-rate sets: sequence mode tracks whole sequences.
SEQUENCE_SETS = ("val-full", "test-dev-full")


@dataclass(frozen=True)
class Image:
    """One frame's ground truth, boxes as ``(n, 4)`` xywh source pixels."""

    name: str  # unique across sequences
    path: Path
    width: int
    height: int
    boxes: np.ndarray  # categories 1-10 only
    categories: np.ndarray  # raw category per box
    track_ids: np.ndarray  # VID's target_id per box
    ignored: np.ndarray  # the frame's ignored regions
    sequence: str
    frame: int  # 1-based, as VID numbers it


def read_sequence(
    seq_dir: Path, annotation_path: Path, pick: Callable[[list[int]], list[int]]
) -> list[Image]:
    """The picked frames of one sequence, in frame order."""
    paths = {int(p.stem): p for p in seq_dir.glob("*.jpg")}
    rows = np.loadtxt(annotation_path, delimiter=",", usecols=range(10), ndmin=2, dtype=np.int64)
    with PILImage.open(paths[min(paths)]) as im:  # one size per sequence, as the toolkit reads it
        width, height = im.size
    images = []
    for f in pick(sorted(paths)):
        frame = rows[rows[:, FRAME_COL] == f]
        category = frame[:, CATEGORY_COL]
        scored = np.isin(category, list(CLASS_MERGE))
        images.append(
            Image(
                name=f"{frame_name(seq_dir.name, f)}.jpg",
                path=paths[f],
                width=width,
                height=height,
                boxes=frame[scored, BOX_COLS].astype(float),
                categories=category[scored],
                track_ids=frame[scored, TRACK_COL],
                ignored=frame[category == IGNORED_CATEGORY, BOX_COLS],
                sequence=seq_dir.name,
                frame=f,
            )
        )
    return images


def read_set(raw_dir: Path, scoring_set: str) -> list[Image]:
    """Every frame of a scoring set, sorted by name."""
    split, pick = SETS[scoring_set]
    root = raw_dir / raw_split_dir(split)
    return [
        im
        for seq in sorted((root / "sequences").iterdir())
        for im in read_sequence(seq, root / "annotations" / f"{seq.name}.txt", pick)
    ]


def outside_ignored(boxes: np.ndarray, ignored: np.ndarray, width: int, height: int) -> np.ndarray:
    """``dropObjectsInIgr.m``: keep a box unless half its w*h lies in the union of ``ignored``.

    Both are ``(n, 4)`` xywh. The toolkit's 1-based inclusive pixel ranges and its rounding
    (half away from zero) are kept, and the denominator is the box's own unclipped area.
    """
    keep = np.ones(len(boxes), bool)
    if not len(boxes) or not len(ignored):
        return keep
    union = np.zeros((height, width), np.uint8)
    for x, y, w, h in np.maximum(1, ignored).astype(int):
        union[y - 1 : min(height, y + h), x - 1 : min(width, x + w)] = 1
    for i, (x, y, w, h) in enumerate(np.maximum(1, np.floor(boxes + 0.5)).astype(int)):
        x, y = min(width, x), min(height, y)
        covered = union[y : min(height, y + h), x : min(width, x + w)].sum()
        keep[i] = covered / (w * h) < IGNORED_SHARE
    return keep


def scored(image: Image) -> Image:
    """``image`` with the ground truth the ignored-region rule drops removed."""
    keep = outside_ignored(image.boxes, image.ignored, image.width, image.height)
    return replace(
        image,
        boxes=image.boxes[keep],
        categories=image.categories[keep],
        track_ids=image.track_ids[keep],
    )


def merged(categories: np.ndarray) -> np.ndarray:
    return np.array([CLASS_MERGE[int(c)] for c in categories], int)


def yolo_lines(image: Image) -> list[str]:
    """Merged-class YOLO lines, the box corners clipped to the image."""
    x1, y1 = image.boxes[:, 0], image.boxes[:, 1]
    x2, y2 = x1 + image.boxes[:, 2], y1 + image.boxes[:, 3]
    x1, x2 = np.clip([x1, x2], 0, image.width) / image.width
    y1, y2 = np.clip([y1, y2], 0, image.height) / image.height
    return [
        f"{c} {(a + b) / 2:.6f} {(t + u) / 2:.6f} {b - a:.6f} {u - t:.6f}"
        for c, a, t, b, u in zip(merged(image.categories), x1, y1, x2, y2, strict=True)
        if b > a and u > t
    ]


def coco_images(images: list[Image]) -> list[dict[str, Any]]:
    """COCO ``images``: ids follow name order, so every file of a scoring set shares them."""
    return [
        {"id": i, "file_name": im.name, "width": im.width, "height": im.height}
        for i, im in enumerate(images, start=1)
    ]


COCO_CATEGORIES = [{"id": c + 1, "name": n} for c, n in YOLO_NAMES.items()]


def coco_ground_truth(images: list[Image]) -> dict[str, Any]:
    """Merged-class COCO ground truth with VID's ``track_id``, the raw category as an attribute."""
    annotations = [
        {
            "image_id": image_id,
            "category_id": CLASS_MERGE[int(c)] + 1,
            "bbox": box.tolist(),
            "area": float(box[2] * box[3]),
            "iscrowd": 0,
            "track_id": int(t),
            "attributes": {"raw_category": RAW_NAMES[int(c)]},
        }
        for image_id, im in enumerate(images, start=1)
        for box, c, t in zip(im.boxes, im.categories, im.track_ids, strict=True)
    ]
    return coco_file(images, annotations, COCO_CATEGORIES)


def coco_ignored(images: list[Image]) -> dict[str, Any]:
    annotations = [
        {
            "image_id": image_id,
            "category_id": 1,
            "bbox": box.tolist(),
            "area": float(box[2] * box[3]),
            "iscrowd": 0,
        }
        for image_id, im in enumerate(images, start=1)
        for box in im.ignored
    ]
    return coco_file(images, annotations, [{"id": 1, "name": "ignored"}])


def coco_file(
    images: list[Image], annotations: list[dict[str, Any]], categories: list[dict[str, Any]]
) -> dict[str, Any]:
    for i, a in enumerate(annotations, start=1):
        a["id"] = i
    return {"images": coco_images(images), "annotations": annotations, "categories": categories}


def scoring_dir(autolabel_dir: Path, scoring_set: str) -> Path:
    return autolabel_dir / scoring_set


def prepare(raw_dir: Path, autolabel_dir: Path, scoring_set: str) -> list[Image]:
    """Write a scoring set's dataset; returns its frames with the ground truth as scored."""
    raw = read_set(raw_dir, scoring_set)
    images = [scored(im) for im in raw]
    out = scoring_dir(autolabel_dir, scoring_set)
    for sub in ("images", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for im in images:
        # One link per file: ultralytics resolves a linked directory and would label the raw one.
        link = out / "images" / im.name
        if not link.is_symlink():
            link.symlink_to(im.path.resolve())
        (out / "labels" / Path(im.name).with_suffix(".txt")).write_text("\n".join(yolo_lines(im)))
    (out / "labels.cache").unlink(missing_ok=True)
    splits = dict.fromkeys(("train", "val", "test"), "images")
    data = {"path": str(out.resolve()), **splits, "names": YOLO_NAMES}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    (out / "ground_truth.json").write_text(json.dumps(coco_ground_truth(images)))
    (out / "ignored.json").write_text(json.dumps(coco_ignored(images)))
    dropped = sum(len(r.boxes) - len(s.boxes) for r, s in zip(raw, images, strict=True))
    logger.info(
        f"{scoring_set}: {len(images)} frames, {sum(len(im.boxes) for im in images)} scored boxes, "
        f"{dropped} dropped by the ignored-region rule -> {out}"
    )
    return images


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write the VisDrone-VID scoring sets.")
    p.add_argument("--set", choices=SETS, action="append", dest="sets", help="default: every set")
    args = p.parse_args(argv)

    from mlops_cv.config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    for scoring_set in args.sets or SETS:
        prepare(settings.data.raw_dir, settings.data.autolabel_dir, scoring_set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
