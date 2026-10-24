"""Slice the subset's train frames for slicing-aided fine-tuning, SAHI's training half.

The windows are the ones ``sahi_eval``'s sliced inference uses (imgsz squares, overlap 0.2).
Slices land in ``images/train-sliced<imgsz>`` (labels alongside); the full frames stay in
``images/train`` and train with them
(``configs/tuning/yolo26-visdrone-vid/datasets/VisDrone-VID-merged-sliced1280.yaml``).
"""

from __future__ import annotations

import argparse
import logging
import shutil

import numpy as np

from mlops_cv.data.subset import IMAGES_DIRNAME, image_relpath, label_relpath
from mlops_cv.tuning.sahi_eval import OVERLAP

logger = logging.getLogger(__name__)

MIN_AREA_RATIO = 0.1  # sahi.slicing's default: a box stays if a tenth of it is inside the window
JPEG_QUALITY = 95


def slice_labels(
    labels: np.ndarray, size: tuple[int, int], window: tuple[int, int, int, int]
) -> np.ndarray:
    """YOLO rows ``(cls, cx, cy, w, h)`` normalised to a ``size`` frame -> rows normalised to
    ``window`` (``x0, y0, x1, y1`` px), clipped to it, keeping boxes with ``MIN_AREA_RATIO`` in."""
    fw, fh = size
    x0, y0, x1, y1 = window
    cls, cx, cy, bw, bh = labels.reshape(-1, 5).T
    bx0, bx1 = (cx - bw / 2) * fw, (cx + bw / 2) * fw
    by0, by1 = (cy - bh / 2) * fh, (cy + bh / 2) * fh
    cx0, cx1 = np.clip(bx0, x0, x1), np.clip(bx1, x0, x1)
    cy0, cy1 = np.clip(by0, y0, y1), np.clip(by1, y0, y1)
    inside = (cx1 - cx0) * (cy1 - cy0)
    keep = (inside > 0) & (inside >= MIN_AREA_RATIO * (bx1 - bx0) * (by1 - by0))
    ww, wh = x1 - x0, y1 - y0
    rows = np.stack(
        [cls, (cx0 + cx1) / 2 - x0, (cy0 + cy1) / 2 - y0, cx1 - cx0, cy1 - cy0], axis=1
    ) / [1, ww, wh, ww, wh]
    return rows[keep]


def main(argv: list[str] | None = None) -> int:
    from PIL import Image
    from sahi.slicing import get_slice_bboxes

    from mlops_cv.config import get_settings

    p = argparse.ArgumentParser(description="Slice the subset's train frames into imgsz windows.")
    p.add_argument("--imgsz", type=int, default=1280)
    args = p.parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")

    root = settings.data.subset_dir
    split = f"train-sliced{args.imgsz}"
    for out in (root / image_relpath(split, "x")).parent, (root / label_relpath(split, "x")).parent:
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)

    frames = sorted((root / IMAGES_DIRNAME / "train").glob("*.jpg"))
    n = 0
    for path in frames:
        text = (root / label_relpath("train", path.stem)).read_text(encoding="utf-8")
        labels = np.array(text.split(), np.float64)
        with Image.open(path) as image:
            w, h = image.size
            windows = get_slice_bboxes(h, w, args.imgsz, args.imgsz, False, OVERLAP, OVERLAP)
            for x0, y0, x1, y1 in windows:
                name = f"{path.stem}_{x0}_{y0}"
                image.crop((x0, y0, x1, y1)).save(
                    root / image_relpath(split, name), quality=JPEG_QUALITY
                )
                rows = slice_labels(labels, (w, h), (x0, y0, x1, y1))
                (root / label_relpath(split, name)).write_text(
                    "".join(
                        f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n" for r in rows
                    ),
                    encoding="utf-8",
                )
                n += 1
    logger.info(f"{len(frames)} train frames -> {n} slices in {root / IMAGES_DIRNAME / split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
