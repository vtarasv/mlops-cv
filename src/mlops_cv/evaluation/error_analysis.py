"""Qualitative error analysis: the lowest-confidence true positives and highest-confidence false
positives, saved as padded annotated crops + a CSV manifest for MLflow."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from mlops_cv.data.convert_visdrone_vid import read_yolo_labels
from mlops_cv.evaluation.visualize import (
    GT_COLOR,
    PRED_COLOR,
    DrawBox,
    _label_font_size,
    _line_width,
    _load_font,
    draw_boxes,
    yolo_to_pixel,
)

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixel corner-form


@dataclass(frozen=True)
class Detection:
    """One model prediction: class id, confidence, and a pixel corner-form box."""

    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def box(self) -> Box:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class GroundTruth:
    """One ground-truth box: class id and a pixel corner-form box."""

    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def box(self) -> Box:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class Matched:
    """A prediction tagged TP/FP after greedy matching, with its best same-class IoU and source."""

    det: Detection
    is_tp: bool
    iou: float  # best IoU vs the matched (TP) or best-available same-class (FP) GT; 0.0 if none
    image: str  # source image filename, for pooling across the whole split


@dataclass(frozen=True)
class SegmentRef:
    """A written crop: its bucket, provenance, and stats (one row of ``segments.csv``)."""

    kind: str  # "low_conf_tp" | "high_conf_fp"
    image: str
    cls_name: str
    conf: float
    iou: float
    crop_path: str  # relative to the analysis output dir


@dataclass(frozen=True)
class ErrorAnalysisResult:
    """The selected crops + the manifest path + simple coverage counts."""

    low_conf_tp: list[SegmentRef]
    high_conf_fp: list[SegmentRef]
    csv_path: Path
    n_images: int
    n_preds: int


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes; 0.0 if disjoint or empty."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def match_detections(
    preds: Sequence[Detection],
    gts: Sequence[GroundTruth],
    iou_thr: float,
    *,
    image: str = "",
) -> list[Matched]:
    """Greedy per-class TP/FP labelling of one image's predictions against its ground truth.

    Predictions are claimed in descending-confidence order; each takes the unused same-class GT of
    highest IoU. ``IoU >= iou_thr`` → TP (that GT is consumed), otherwise → FP (covering both "no
    overlap" and "duplicate of an already-claimed GT"). Output preserves input order.
    """
    used = [False] * len(gts)
    tagged: dict[int, tuple[bool, float]] = {}
    for i in sorted(range(len(preds)), key=lambda idx: (-preds[idx].conf, idx)):
        det = preds[i]
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if used[j] or gt.cls != det.cls:
                continue
            v = iou(det.box, gt.box)
            if v > best_iou:
                best_iou, best_j = v, j
        is_tp = best_j >= 0 and best_iou >= iou_thr
        if is_tp:
            used[best_j] = True
        tagged[i] = (is_tp, best_iou)
    return [Matched(preds[i], tagged[i][0], tagged[i][1], image) for i in range(len(preds))]


def select_segments(matched: Sequence[Matched], k: int) -> tuple[list[Matched], list[Matched]]:
    """From a pooled ``Matched`` list over the whole split, the ``k`` lowest-confidence TPs and the
    ``k`` highest-confidence FPs; fewer than ``k`` is fine.
    """
    tps = [m for m in matched if m.is_tp]
    fps = [m for m in matched if not m.is_tp]
    low_conf_tp = sorted(tps, key=lambda m: (m.det.conf, m.image))[:k]
    high_conf_fp = sorted(fps, key=lambda m: (-m.det.conf, m.image))[:k]
    return low_conf_tp, high_conf_fp


def _crop_box(det: Detection, width: int, height: int) -> tuple[int, int, int, int]:
    """A padded, in-bounds crop region around ``det`` (≥16 px or 20% of the box per side)."""
    pad_x = max(0.2 * (det.x2 - det.x1), 16.0)
    pad_y = max(0.2 * (det.y2 - det.y1), 16.0)
    x0 = max(0, int(det.x1 - pad_x))
    y0 = max(0, int(det.y1 - pad_y))
    x1 = min(width, int(det.x2 + pad_x))
    y1 = min(height, int(det.y2 + pad_y))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def _render_bucket(
    picks: Sequence[Matched],
    kind: str,
    color: tuple[int, int, int],
    names: Mapping[int, str],
    images_dir: Path,
    out_dir: Path,
) -> list[SegmentRef]:
    bucket_dir = out_dir / kind
    bucket_dir.mkdir(parents=True, exist_ok=True)
    refs: list[SegmentRef] = []
    for rank, m in enumerate(picks, start=1):
        det = m.det
        cls_name = names.get(det.cls, str(det.cls))
        with Image.open(images_dir / m.image) as im:
            width, height = im.size
            crop_box = _crop_box(det, width, height)
            crop = im.convert("RGB").crop(crop_box)
        x0, y0 = crop_box[0], crop_box[1]
        local = DrawBox(det.x1 - x0, det.y1 - y0, det.x2 - x0, det.y2 - y0, cls_name, det.conf)
        font = _load_font(_label_font_size(crop.height))
        draw_boxes(crop, [local], color, width=_line_width(crop.height), font=font, with_conf=True)
        rel = f"{kind}/{rank:02d}_{Path(m.image).stem}_{cls_name}.jpg"
        crop.save(out_dir / rel)
        refs.append(SegmentRef(kind, m.image, cls_name, det.conf, m.iou, rel))
    return refs


def _detections(result: object) -> list[Detection]:
    boxes = result.boxes  # type: ignore[attr-defined]
    if boxes is None:
        return []
    return [
        Detection(int(c), float(cf), float(x1), float(y1), float(x2), float(y2))
        for (x1, y1, x2, y2), c, cf in zip(
            boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), strict=True
        )
    ]


def run_error_analysis(
    model: object,
    images_dir: str | Path,
    labels_dir: str | Path,
    names: Mapping[int, str],
    out_dir: str | Path,
    *,
    k: int = 10,
    conf: float = 0.25,
    iou_thr: float = 0.5,
) -> ErrorAnalysisResult:
    """Predict on every image, greedily match vs the paired YOLO label, then write the global
    lowest-confidence-TP / highest-confidence-FP crops + ``segments.csv``. Predicts one image at a
    time (batch 1); ``model`` needs only ``.predict(src, conf=, verbose=)``.
    """
    images_dir, labels_dir, out_dir = Path(images_dir), Path(labels_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool: list[Matched] = []
    n_preds = 0
    image_paths = sorted(images_dir.glob("*.jpg"))
    for fp in image_paths:
        result = model.predict(str(fp), conf=conf, verbose=False)[0]  # type: ignore[attr-defined]
        preds = _detections(result)
        n_preds += len(preds)
        height, width = result.orig_shape  # type: ignore[attr-defined]  # ultralytics: (h, w)
        gts = [
            GroundTruth(b.cls, *_yolo_corners(b, width, height))
            for b in read_yolo_labels(labels_dir / f"{fp.stem}.txt")
        ]
        pool.extend(match_detections(preds, gts, iou_thr, image=fp.name))

    low, high = select_segments(pool, k)
    refs_low = _render_bucket(low, "low_conf_tp", GT_COLOR, names, images_dir, out_dir)
    refs_high = _render_bucket(high, "high_conf_fp", PRED_COLOR, names, images_dir, out_dir)

    csv_path = out_dir / "segments.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "image", "class", "conf", "iou", "crop_path"])
        for ref in [*refs_low, *refs_high]:
            writer.writerow(
                [
                    ref.kind,
                    ref.image,
                    ref.cls_name,
                    f"{ref.conf:.4f}",
                    f"{ref.iou:.4f}",
                    ref.crop_path,
                ]
            )

    return ErrorAnalysisResult(refs_low, refs_high, csv_path, len(image_paths), n_preds)


def _yolo_corners(box: object, width: int, height: int) -> Box:
    px = yolo_to_pixel(box, width, height)  # type: ignore[arg-type]
    return px.x1, px.y1, px.x2, px.y2
