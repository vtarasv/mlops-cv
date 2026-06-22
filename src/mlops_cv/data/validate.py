"""Validate a YOLO detection dataset, failing fast on schema/value skew."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mlops_cv.config import get_settings
from mlops_cv.data.convert_visdrone_vid import YOLO_NAMES

SPLITS = ("train", "val")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_CLASSES = set(YOLO_NAMES)


class DatasetValidationError(Exception):
    """Raised when a dataset fails one or more validation checks."""


@dataclass(frozen=True)
class Report:
    """Summary of a passing dataset validation."""

    splits: dict[str, int]  # split -> image count
    class_counts: dict[int, int]  # merged class id -> box count
    n_images: int
    n_labels: int
    n_boxes: int

    def summary(self) -> str:
        per_split = ", ".join(f"{s}={self.splits.get(s, 0)}" for s in SPLITS)
        per_class = ", ".join(
            f"{YOLO_NAMES[c]}={self.class_counts.get(c, 0)}" for c in sorted(YOLO_NAMES)
        )
        return f"OK: {self.n_images} images ({per_split}), {self.n_boxes} boxes ({per_class})"


def _image_stems(images_dir: Path) -> set[str]:
    if not images_dir.is_dir():
        return set()
    return {p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS}


def _label_paths(labels_dir: Path) -> list[Path]:
    return sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []


def validate_dataset(
    dataset_dir: str | Path,
    *,
    class_bounds: Mapping[int, tuple[int, int | None]] | None = None,
) -> Report:
    """Validate the YOLO dataset under ``dataset_dir``; raise on any failure, else return a Report.

    Checks: train & val splits non-empty; exact image<->label pairing per split; every class id in
    ``{0,1,2}``; every bbox coord in [0, 1] with positive width/height; and each merged class within
    ``class_bounds`` (default: at least 1 box each).
    """
    root = Path(dataset_dir)
    errors: list[str] = []
    splits_count: dict[str, int] = {}
    class_counts: dict[int, int] = dict.fromkeys(ALLOWED_CLASSES, 0)
    n_images = n_labels = n_boxes = 0

    for split in SPLITS:
        images_dir, labels_dir = root / "images" / split, root / "labels" / split
        img_stems = _image_stems(images_dir)
        label_paths = _label_paths(labels_dir)
        label_stems = {p.stem for p in label_paths}
        splits_count[split] = len(img_stems)
        n_images += len(img_stems)
        n_labels += len(label_paths)

        if not img_stems:
            errors.append(f"[{split}] no images found under {images_dir}")
        if missing := sorted(img_stems - label_stems):
            errors.append(f"[{split}] {len(missing)} image(s) without a label, e.g. {missing[:3]}")
        if orphan := sorted(label_stems - img_stems):
            errors.append(f"[{split}] {len(orphan)} label(s) without an image, e.g. {orphan[:3]}")

        for lp in label_paths:
            for n, raw in enumerate(lp.read_text(encoding="utf-8").splitlines(), start=1):
                fields = raw.split()
                if not fields:
                    continue
                if len(fields) != 5:
                    errors.append(f"[{split}] {lp.name}:{n} expected 5 fields, got {len(fields)}")
                    continue
                try:
                    cls, coords = int(fields[0]), [float(x) for x in fields[1:]]
                except ValueError:
                    errors.append(f"[{split}] {lp.name}:{n} non-numeric token: {raw!r}")
                    continue
                n_boxes += 1
                if cls in ALLOWED_CLASSES:
                    class_counts[cls] += 1
                else:
                    errors.append(
                        f"[{split}] {lp.name}:{n} class id {cls} not in {sorted(ALLOWED_CLASSES)}"
                    )
                if not all(0.0 <= v <= 1.0 for v in coords):
                    errors.append(f"[{split}] {lp.name}:{n} bbox out of [0,1]: {coords}")
                elif coords[2] <= 0.0 or coords[3] <= 0.0:
                    errors.append(f"[{split}] {lp.name}:{n} non-positive box size: {coords[2:]}")

    bounds = class_bounds or {c: (1, None) for c in ALLOWED_CLASSES}
    for cls, (lo, hi) in bounds.items():
        count = class_counts.get(cls, 0)
        name = YOLO_NAMES.get(cls, cls)
        if count < lo:
            errors.append(f"class {name} has {count} boxes, expected >= {lo}")
        if hi is not None and count > hi:
            errors.append(f"class {name} has {count} boxes, expected <= {hi}")

    if errors:
        raise DatasetValidationError(
            f"{len(errors)} dataset validation error(s):\n  - " + "\n  - ".join(errors)
        )
    return Report(splits_count, class_counts, n_images, n_labels, n_boxes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a YOLO dataset built by build_subset.")
    parser.add_argument(
        "--dataset-dir", type=Path, default=None, help="defaults to settings.data.subset_dir"
    )
    args = parser.parse_args(argv)
    dataset_dir = args.dataset_dir or get_settings().data.subset_dir
    try:
        report = validate_dataset(dataset_dir)
    except DatasetValidationError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
