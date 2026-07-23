"""The subset's on-disk shape: layout vocabulary, path constructors, manifest IO, YAML stamping.

Single owner of how a training subset is laid out on disk. Producers (the ingest pipeline)
and consumers (validation, profiling, evaluation, orchestration) build every subset path
through this module; path constructors return POSIX-style relative strings so callers can
join them with their own filesystem dialect (``pathlib`` locally, Beam ``FileSystems``
in pipelines).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from pathlib import Path

import yaml

MANIFEST_FIELDS = [
    "split",
    "sequence",
    "frame_index",
    "image_relpath",
    "label_relpath",
    "width",
    "height",
    "n_boxes",
    "n_person",
    "n_vehicle",
    "n_two_three_wheeler",
]
MANIFEST_FILENAME = "manifest.csv"

# YOLO split name -> raw VisDrone-VID directory.
SPLITS = ("train", "val", "test")
RAW_SPLIT_DIRS = {
    "train": "VisDrone2019-VID-train",
    "val": "VisDrone2019-VID-val",
    "test": "VisDrone2019-VID-test-dev",
}

IMAGES_DIRNAME = "images"
LABELS_DIRNAME = "labels"
# Subset subdir holding full-rate demo-clip frames + labels (rendered by evaluation).
DEMO_DIRNAME = "demo"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def raw_split_dir(split: str) -> str:
    """Raw VisDrone-VID directory name for a YOLO split name."""
    return RAW_SPLIT_DIRS[split]


def frame_name(sequence: str, frame_index: int) -> str:
    """Flattened, sortable stem of one subset frame: ``<sequence>_<0000000-padded index>``."""
    return f"{sequence}_{frame_index:07d}"


def image_relpath(split: str, name: str) -> str:
    """Subset-relative path of a frame image, e.g. ``images/train/<name>.jpg``."""
    return f"{IMAGES_DIRNAME}/{split}/{name}.jpg"


def label_relpath(split: str, name: str) -> str:
    """Subset-relative path of a frame's YOLO label file, e.g. ``labels/train/<name>.txt``."""
    return f"{LABELS_DIRNAME}/{split}/{name}.txt"


def demo_image_relpath(sequence: str, frame_index: int) -> str:
    """Subset-relative path of a demo-store frame image (full frame rate, per-sequence dirs)."""
    return f"{DEMO_DIRNAME}/{IMAGES_DIRNAME}/{sequence}/{frame_index:07d}.jpg"


def demo_label_relpath(sequence: str, frame_index: int) -> str:
    """Subset-relative path of a demo-store frame's ground-truth label file."""
    return f"{DEMO_DIRNAME}/{LABELS_DIRNAME}/{sequence}/{frame_index:07d}.txt"


def demo_store_present(subset_dir: str | Path) -> bool:
    """True iff the subset carries at least one demo-store frame."""
    images = Path(subset_dir) / DEMO_DIRNAME / IMAGES_DIRNAME
    return any(images.glob("*/*.jpg"))


def manifest_header() -> str:
    """The manifest's CSV header line."""
    return ",".join(MANIFEST_FIELDS)


def manifest_csv_line(row: dict[str, object]) -> str:
    """One manifest row as a CSV line (no trailing newline), field order = ``MANIFEST_FIELDS``."""
    buf = io.StringIO()
    csv.writer(buf).writerow([row[field] for field in MANIFEST_FIELDS])
    return buf.getvalue().rstrip("\r\n")


def manifest_rows(lines: Iterable[str]) -> list[dict[str, str]]:
    """Parse manifest CSV text lines into row dicts, validating the header."""
    reader = csv.DictReader(lines)
    if reader.fieldnames is None or list(reader.fieldnames) != MANIFEST_FIELDS:
        raise ValueError(
            f"unexpected manifest header {reader.fieldnames} (expected {MANIFEST_FIELDS})"
        )
    return list(reader)


def stamp_dataset_yaml(template: Path, out_dir: Path) -> Path:
    """Copy the template YAML next to the data with ``path`` stamped to the absolute subset dir."""
    data = yaml.safe_load(Path(template).read_text(encoding="utf-8"))
    data["path"] = str(out_dir.resolve())
    dest = out_dir / Path(template).name
    dest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return dest
