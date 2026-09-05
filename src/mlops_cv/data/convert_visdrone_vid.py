"""Convert VisDrone-VID annotations to YOLO labels with a 3-class merge.

VisDrone-VID ground-truth line (one per object per frame, comma-separated, 0-based columns)::

    0:frame_index
    1:target_id
    2:bbox_left
    3:bbox_top
    4:bbox_width
    5:bbox_height
    6:score
    7:object_category
    8:truncation
    9:occlusion

``score`` is 1 (considered) or 0 (ignored region — skipped). bbox is absolute pixels with
(left, top) the top-left corner. The output YOLO line is ``<cls> <xc> <yc> <w> <h>`` with the box
*center* and size normalized to [0, 1] by the image width/height.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# person <- 1,2 ; vehicle <- 4,5,6,9 ; two-three-wheeler <- 3,7,8,10
CLASS_MERGE: dict[int, int] = {1: 0, 2: 0, 4: 1, 5: 1, 6: 1, 9: 1, 3: 2, 7: 2, 8: 2, 10: 2}
YOLO_NAMES: dict[int, str] = {0: "person", 1: "vehicle", 2: "two-three-wheeler"}

# Documented column indices (0-based) used by the converter.
FRAME_COL, LEFT_COL, TOP_COL, WIDTH_COL, HEIGHT_COL, SCORE_COL, CATEGORY_COL = 0, 2, 3, 4, 5, 6, 7


@dataclass(frozen=True)
class YoloBox:
    """A single normalized YOLO box (center + size, all in [0, 1])."""

    cls: int
    xc: float
    yc: float
    w: float
    h: float

    def to_line(self) -> str:
        return f"{self.cls} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"


def _clip01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def convert_annotation_line(line: str, img_w: int, img_h: int) -> tuple[int, YoloBox] | None:
    """Convert one VisDrone-VID annotation line to ``(frame_index, YoloBox)``.

    Returns ``None`` for lines that should be dropped: blank/malformed, ``score == 0`` (ignored
    region), a category not in :data:`CLASS_MERGE` (raw 0/11), or a box that clips to zero area.
    ``img_w``/``img_h`` are plain ints.
    """
    if img_w <= 0 or img_h <= 0:
        return None
    parts = line.strip().split(",")
    if len(parts) < 8:
        return None
    try:
        frame_index = int(parts[FRAME_COL])
        left, top = float(parts[LEFT_COL]), float(parts[TOP_COL])
        width, height = float(parts[WIDTH_COL]), float(parts[HEIGHT_COL])
        score, category = int(parts[SCORE_COL]), int(parts[CATEGORY_COL])
    except ValueError:
        return None
    if score == 0 or category not in CLASS_MERGE:
        return None
    # Clip the box *corners* to the image, then recompute center+size, so a box crossing the frame
    # edge is trimmed correctly (clipping the center alone would not).
    x1, x2 = _clip01(left / img_w), _clip01((left + width) / img_w)
    y1, y2 = _clip01(top / img_h), _clip01((top + height) / img_h)
    nw, nh = x2 - x1, y2 - y1
    if nw <= 0.0 or nh <= 0.0:
        return None
    return frame_index, YoloBox(CLASS_MERGE[category], (x1 + x2) / 2, (y1 + y2) / 2, nw, nh)


def convert_sequence_text(text: str, image_size: tuple[int, int]) -> dict[int, list[YoloBox]]:
    """Convert one sequence's annotation text into ``{frame_index: [YoloBox, ...]}``."""
    img_w, img_h = image_size
    by_frame: dict[int, list[YoloBox]] = {}
    for line in text.splitlines():
        result = convert_annotation_line(line, img_w, img_h)
        if result is None:
            continue
        frame_index, box = result
        by_frame.setdefault(frame_index, []).append(box)
    return by_frame


def image_size(path: str | Path) -> tuple[int, int]:
    """Return an image's ``(width, height)``."""
    with Image.open(path) as im:
        return im.size


def labels_to_text(boxes: list[YoloBox]) -> str:
    """Serialize boxes to YOLO label-file text (empty string for no boxes)."""
    text = "\n".join(box.to_line() for box in boxes)
    return text + "\n" if text else ""


def labels_from_text(text: str) -> list[YoloBox]:
    """Parse YOLO label-file text (``<cls> <xc> <yc> <w> <h>`` per line), skipping malformed
    lines. The inverse of :func:`labels_to_text`.
    """
    boxes: list[YoloBox] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            cls = int(fields[0])
            xc, yc, w, h = (float(v) for v in fields[1:])
        except ValueError:
            continue
        boxes.append(YoloBox(cls, xc, yc, w, h))
    return boxes


def read_yolo_labels(path: str | Path) -> list[YoloBox]:
    """Read a YOLO label file into :class:`YoloBox` es (missing file == zero objects)."""
    label_path = Path(path)
    if not label_path.is_file():
        return []
    return labels_from_text(label_path.read_text(encoding="utf-8"))
