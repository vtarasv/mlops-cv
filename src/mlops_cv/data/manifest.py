"""Dataset schema shared by the batch pipelines: layout constants, manifest IO, YAML stamping."""

from __future__ import annotations

import csv
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

# YOLO split name -> raw VisDrone-VID directory.
SPLITS = ("train", "val", "test")
RAW_SPLIT_DIRS = {
    "train": "VisDrone2019-VID-train",
    "val": "VisDrone2019-VID-val",
    "test": "VisDrone2019-VID-test-dev",
}

# Subset subdir holding full-rate demo-clip frames + labels (rendered by evaluation).
DEMO_DIRNAME = "demo"


def write_manifest(rows: list[dict], dest: Path) -> None:
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def stamp_dataset_yaml(template: Path, out_dir: Path) -> Path:
    """Copy the template YAML next to the data with ``path`` stamped to the absolute subset dir."""
    data = yaml.safe_load(Path(template).read_text(encoding="utf-8"))
    data["path"] = str(out_dir.resolve())
    dest = out_dir / Path(template).name
    dest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return dest
