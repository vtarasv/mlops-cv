"""Raw VisDrone-DET: its annotations, the toolkit's ignored-region rule and the scoring dataset.

DET annotation row (0-based columns): ``left, top, width, height, score, category, truncation,
occlusion``. Category 0 rows are ignored regions; category 11 (``others``) is never scored.

The scoring dataset is ``<autolabel_dir>/det-<split>/``: ``images`` (a link to the raw images),
3-class YOLO ``labels``, ``data.yaml``, and COCO ``ground_truth.json`` + ``ignored.json`` for the
annotation editor. Ground truth is dropped by the ignored-region rule before it is written.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image as PILImage

from mlops_cv.data.convert_visdrone_vid import CLASS_MERGE, YOLO_NAMES
from mlops_cv.startup import exits_on_startup_error

logger = logging.getLogger(__name__)

SPLITS = {"val": "VisDrone2019-DET-val", "test-dev": "VisDrone2019-DET-test-dev"}
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
CATEGORY_COL = 5
# The toolkit drops a box when at least this share of its w*h lies inside the ignored regions.
IGNORED_SHARE = 0.5
# The split every scoring dataset YAML names: ultralytics' check needs ``val:``.
SCORING_SPLIT = "val"


@dataclass(frozen=True)
class Image:
    """One DET image's ground truth, boxes as ``(n, 4)`` xywh source pixels."""

    name: str
    width: int
    height: int
    boxes: np.ndarray  # categories 1-10 only
    categories: np.ndarray  # raw category per box
    ignored: np.ndarray  # the ignored regions


def read_image(image_path: Path, annotation_path: Path) -> Image:
    rows = np.loadtxt(annotation_path, delimiter=",", usecols=range(8), ndmin=2, dtype=np.int64)
    category = rows[:, CATEGORY_COL]
    scored = np.isin(category, list(CLASS_MERGE))
    with PILImage.open(image_path) as im:
        width, height = im.size
    return Image(
        name=image_path.name,
        width=width,
        height=height,
        boxes=rows[scored, :4].astype(float),
        categories=category[scored],
        ignored=rows[category == IGNORED_CATEGORY, :4],
    )


def split_dir(det_dir: Path, split: str) -> Path:
    return det_dir / SPLITS[split]


def read_split(det_dir: Path, split: str) -> list[Image]:
    """Every image of a raw DET split, sorted by file name."""
    root = split_dir(det_dir, split)
    return [
        read_image(p, root / "annotations" / p.with_suffix(".txt").name)
        for p in sorted((root / "images").glob("*.jpg"))
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
    return replace(image, boxes=image.boxes[keep], categories=image.categories[keep])


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
    """COCO ``images``: ids follow file-name order, so every file of a split shares them."""
    return [
        {"id": i, "file_name": im.name, "width": im.width, "height": im.height}
        for i, im in enumerate(images, start=1)
    ]


COCO_CATEGORIES = [{"id": c + 1, "name": n} for c, n in YOLO_NAMES.items()]


def coco_ground_truth(images: list[Image]) -> dict[str, Any]:
    """Merged-class COCO ground truth, the raw category kept as an attribute."""
    annotations = [
        {
            "image_id": image_id,
            "category_id": CLASS_MERGE[int(c)] + 1,
            "bbox": box.tolist(),
            "area": float(box[2] * box[3]),
            "iscrowd": 0,
            "attributes": {"raw_category": RAW_NAMES[int(c)]},
        }
        for image_id, im in enumerate(images, start=1)
        for box, c in zip(im.boxes, im.categories, strict=True)
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


def scoring_dir(autolabel_dir: Path, split: str) -> Path:
    return autolabel_dir / f"det-{split}"


def prepare(det_dir: Path, autolabel_dir: Path, split: str) -> list[Image]:
    """Write ``split``'s scoring dataset; returns its images with the ground truth as scored."""
    raw = read_split(det_dir, split)
    images = [scored(im) for im in raw]
    out = scoring_dir(autolabel_dir, split)
    for sub in ("images", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    source = (split_dir(det_dir, split) / "images").resolve()
    for im in images:
        # One link per file: ultralytics resolves a linked directory and would label the raw one.
        link = out / "images" / im.name
        if not link.is_symlink():
            link.symlink_to(source / im.name)
        (out / "labels" / Path(im.name).with_suffix(".txt")).write_text("\n".join(yolo_lines(im)))
    (out / "labels.cache").unlink(missing_ok=True)
    splits = dict.fromkeys(("train", "val", "test"), "images")
    data = {"path": str(out.resolve()), **splits, "names": YOLO_NAMES}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    (out / "ground_truth.json").write_text(json.dumps(coco_ground_truth(images)))
    (out / "ignored.json").write_text(json.dumps(coco_ignored(images)))
    dropped = sum(len(r.boxes) - len(s.boxes) for r, s in zip(raw, images, strict=True))
    logger.info(
        f"{split}: {len(images)} images, {sum(len(im.boxes) for im in images)} scored boxes, "
        f"{dropped} dropped by the ignored-region rule -> {out}"
    )
    return images


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write the VisDrone-DET scoring datasets.")
    p.add_argument("--split", choices=SPLITS, action="append", help="default: every split")
    args = p.parse_args(argv)

    from mlops_cv.config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    for split in args.split or SPLITS:
        prepare(settings.data.det_dir, settings.data.autolabel_dir, split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
