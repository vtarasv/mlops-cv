"""Raw VisDrone-VID frames, the toolkit's ignored-region rule, the VisDrone-rule splits and scores.

Ingestion drops the ignored regions, so they are read back from the raw annotations. Each
VisDrone-rule split (``images/<split>-visdrone-rule``, labels alongside) links the subset split's
frames and keeps only the ground truth the rule keeps; labeling trials score against it.
"""

from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image as PILImage

from mlops_cv.data.convert_visdrone_vid import CATEGORY_COL, CLASS_MERGE, FRAME_COL
from mlops_cv.data.subset import (
    IMAGES_DIRNAME,
    MANIFEST_FILENAME,
    frame_name,
    image_relpath,
    label_relpath,
    manifest_rows,
    raw_split_dir,
    stamp_split_yaml,
)
from mlops_cv.startup import exits_on_startup_error

if TYPE_CHECKING:
    from mlops_cv.config import Settings

logger = logging.getLogger(__name__)

IGNORED_CATEGORY = 0
BOX_COLS = slice(2, 6)  # left, top, width, height
TRACK_COL = 1  # target_id
# The toolkit drops a box when at least this share of its w*h lies inside the ignored regions.
IGNORED_SHARE = 0.5
# The subset splits labeling trials score on; test is read once, by YOLO, at the end.
LABEL_SPLITS = ("train", "val")
RULE_SPLITS = (*LABEL_SPLITS, "test")
# The labeler's auto-labels on the subset's train frames, through the rule (``autolabels.py``).
AUTOLABEL_SPLIT = "train-autolabels"
# mAP at IoU 0.75, reported beside a labeling trial's mAP50-95.
MAP75 = "mAP75"
# The confidence at the validator's max-F1 point, where it reads P and R.
CONF_AT_MAX_F1 = "conf_at_max_F1"
# Where P and R read at a fixed confidence (a labeling config's floor) live under a prefix.
AT_FLOOR = "at_floor"


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


def visdrone_rule_split(split: str) -> str:
    """The split's copy under VisDrone's ignored-region rule, e.g. ``train-visdrone-rule``."""
    return f"{split}-visdrone-rule"


def read_sequence(
    seq_dir: Path, annotation_path: Path, wanted: Collection[int] | None = None
) -> list[Image]:
    """The ``wanted`` frames of one sequence (default: all), in frame order."""
    paths = {int(p.stem): p for p in seq_dir.glob("*.jpg")}
    rows = np.loadtxt(annotation_path, delimiter=",", usecols=range(10), ndmin=2, dtype=np.int64)
    with PILImage.open(paths[min(paths)]) as im:  # one size per sequence, as the toolkit reads it
        width, height = im.size
    images = []
    for f in sorted(paths if wanted is None else set(wanted) & paths.keys()):
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


def subset_frames(subset_dir: Path, split: str) -> dict[str, list[int]]:
    """``{sequence: frame indices}`` of one subset split, from its manifest."""
    with (subset_dir / MANIFEST_FILENAME).open(encoding="utf-8") as f:
        rows = manifest_rows(f)
    frames: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            frames[row["sequence"]].append(int(row["frame_index"]))
    return dict(frames)


def read_split(raw_dir: Path, subset_dir: Path, split: str, clips: bool = False) -> list[Image]:
    """A subset split's frames, linked to the subset's images, ground truth through the rule.

    With ``clips``, every raw frame of the split's sequences instead, at their raw paths.
    """
    root = raw_dir / raw_split_dir(split)
    images = []
    for seq, wanted in sorted(subset_frames(subset_dir, split).items()):
        frames = None if clips else wanted
        annotations = root / "annotations" / f"{seq}.txt"
        for im in read_sequence(root / "sequences" / seq, annotations, frames):
            path = im.path if clips else subset_dir / image_relpath(split, Path(im.name).stem)
            images.append(scored(replace(im, path=path)))
    return images


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


def drop_ignored(rows: np.ndarray, image: Image) -> np.ndarray:
    """Predicted ``rows`` (x1 y1 x2 y2 ...) without those the ignored-region rule drops."""
    xywh = np.column_stack([rows[:, :2], rows[:, 2:4] - rows[:, :2]])
    return rows[outside_ignored(xywh, image.ignored, image.width, image.height)]


def drop_all_ignored(
    detections: Mapping[str, np.ndarray], images: list[Image]
) -> dict[str, np.ndarray]:
    """``{frame file name: rows}`` through the rule, for every image."""
    return {im.name: drop_ignored(detections[im.name], im) for im in images}


def readings(
    detections: Mapping[str, np.ndarray],
    data_yaml: Path,
    split: str,
    prefix: str,
    frames: Collection[str] | None = None,
    floor: float | None = None,
) -> dict[str, float]:
    """``common.readings`` plus mAP75 and the max-F1 confidence; with ``floor``, P and R there."""
    from ultralytics.utils.metrics import smooth

    from mlops_cv.evaluation.score_detections import score_with_ap_small
    from mlops_cv.tracking.metric_keys import (
        MAP_SMALL,
        PRIMARY,
        headline_metrics,
        metric_key,
        per_class_metrics,
    )

    # ``common.readings``' keys, assembled here: the extras need the validator's metrics.
    metrics, ap_s = score_with_ap_small(detections, data_yaml, split, frames)
    box = metrics.box
    # ultralytics' own max-F1 index (``ap_per_class``), mapped onto its confidence axis.
    conf = float(box.px[smooth(box.f1_curve.mean(0), 0.1).argmax()]) if len(box.f1_curve) else 0.0
    at_floor: dict[str, float] = {}
    if floor is not None and len(box.p_curve):
        i = np.abs(box.px - floor).argmin()
        at = metric_key(prefix, AT_FLOOR)
        at_floor = {
            metric_key(at, "precision"): float(box.p_curve[:, i].mean()),
            metric_key(at, "recall"): float(box.r_curve[:, i].mean()),
        }
    return (
        headline_metrics(box, prefix=prefix)
        | {metric_key(prefix, MAP75): float(box.map75), metric_key(prefix, MAP_SMALL): ap_s}
        | {metric_key(prefix, CONF_AT_MAX_F1): conf}
        | per_class_metrics(
            metrics.maps,
            metrics.ap_class_index,
            metrics.names,
            prefix=metric_key(prefix, PRIMARY),
        )
        | at_floor
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


def rule_yaml(settings: Settings, split: str) -> Path:
    """The stamped dataset YAML naming only ``split``'s VisDrone-rule copy (scored as ``val``)."""
    subset_dir = settings.data.subset_dir.resolve()
    return stamp_split_yaml(settings.data.dataset_yaml, subset_dir, visdrone_rule_split(split))


def write_split(subset_dir: Path, split: str, source: str, labels: Mapping[str, list[str]]) -> None:
    """Write ``split``: a link to each ``source`` frame named in ``labels``, with its YOLO lines."""
    images_dir = (subset_dir / image_relpath(split, "x")).parent
    labels_dir = (subset_dir / label_relpath(split, "x")).parent
    for out in images_dir, labels_dir:
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
    labels_dir.with_suffix(".cache").unlink(missing_ok=True)
    for name, lines in labels.items():
        # One relative link per file: ultralytics resolves a linked split directory.
        (images_dir / name).symlink_to(Path("..") / source / name)
        (labels_dir / Path(name).with_suffix(".txt")).write_text("".join(f"{x}\n" for x in lines))


def build_rule_split(raw_dir: Path, subset_dir: Path, split: str) -> tuple[int, int]:
    """Write ``split``'s VisDrone-rule copy -> (ground-truth boxes kept, boxes the rule drops)."""
    labels, total = {}, 0
    for im in read_split(raw_dir, subset_dir, split):
        labels[im.name] = yolo_lines(im)
        text = (subset_dir / label_relpath(split, Path(im.name).stem)).read_text(encoding="utf-8")
        total += len(text.splitlines())
    write_split(subset_dir, visdrone_rule_split(split), split, labels)
    kept = sum(map(len, labels.values()))
    return kept, total - kept


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    from mlops_cv.config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    subset_dir = settings.data.subset_dir
    for split in RULE_SPLITS:
        kept, dropped = build_rule_split(settings.data.raw_dir, subset_dir, split)
        rule = subset_dir / IMAGES_DIRNAME / visdrone_rule_split(split)
        logger.info(f"{split}: {kept} boxes kept, {dropped} dropped by the rule -> {rule}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
