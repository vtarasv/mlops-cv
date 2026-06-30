"""Build a small YOLO dataset from VisDrone-VID: all sequences, frame-strided.

Keeps **every** train/val/test sequence (max scene diversity, guaranteed class coverage) and
shrinks the set by ``--frame-stride`` (keep every Nth 1-based frame), which removes
near-duplicate consecutive video frames. Copies (or symlinks) the kept frames into
``images/{train,val,test}/`` with flattened ``<seq>_<NNNNNNN>.jpg`` names, writes matching YOLO
``labels/...``, a ``manifest.csv``, and an **absolute-path** copy of the dataset YAML.

Run as a CLI:
    DATA__RAW_DIR=/path/to/VisDrone-VID uv run python -m mlops_cv.data.build_subset
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path

import yaml

from mlops_cv.config import get_settings
from mlops_cv.data.convert_visdrone_vid import convert_sequence, image_size, write_label_file

logger = logging.getLogger(__name__)

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


def list_sequences(raw_dir: Path, split: str, only: list[str] | None = None) -> list[str]:
    """Sorted sequence names under ``VisDrone2019-VID-<split>/sequences/`` (optionally filtered)."""
    seq_dir = raw_dir / RAW_SPLIT_DIRS[split] / "sequences"
    if not seq_dir.is_dir():
        raise FileNotFoundError(
            f"VisDrone-VID {split} sequences not found at {seq_dir}. Download the dataset "
            "(see docs/data.md) and point DATA__RAW_DIR at the directory holding "
            "VisDrone2019-VID-{train,val,test-dev}/."
        )
    available = sorted(p.name for p in seq_dir.iterdir() if p.is_dir())
    if only is None:
        return available
    wanted = set(only)
    return [s for s in available if s in wanted]


def _place(src: Path, dst: Path, link: bool) -> None:
    """Copy (default) or symlink ``src`` to ``dst``, replacing any existing file."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link:
        dst.symlink_to(src.resolve())
    else:
        shutil.copyfile(src, dst)


def build_split(
    raw_dir: Path, out_dir: Path, split: str, sequences: list[str], stride: int, link: bool
) -> list[dict]:
    """Convert + place the strided, non-empty frames of ``sequences``; return manifest rows."""
    base = raw_dir / RAW_SPLIT_DIRS[split]
    img_out, lbl_out = out_dir / "images" / split, out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for seq in sequences:
        frames = {int(p.stem): p for p in (base / "sequences" / seq).glob("*.jpg")}
        if not frames:
            logger.warning("sequence %s has no frames; skipping", seq)
            continue
        w, h = image_size(frames[min(frames)])  # size is constant within a VisDrone-VID sequence
        by_frame = convert_sequence(base / "annotations" / f"{seq}.txt", (w, h))
        for idx in sorted(by_frame):  # only frames that have surviving boxes
            if (idx - 1) % stride != 0 or idx not in frames:
                continue
            name = f"{seq}_{idx:07d}"
            _place(frames[idx], img_out / f"{name}.jpg", link)
            write_label_file(by_frame[idx], lbl_out / f"{name}.txt")
            counts = {0: 0, 1: 0, 2: 0}
            for box in by_frame[idx]:
                counts[box.cls] += 1
            rows.append(
                {
                    "split": split,
                    "sequence": seq,
                    "frame_index": idx,
                    "image_relpath": f"images/{split}/{name}.jpg",
                    "label_relpath": f"labels/{split}/{name}.txt",
                    "width": w,
                    "height": h,
                    "n_boxes": len(by_frame[idx]),
                    "n_person": counts[0],
                    "n_vehicle": counts[1],
                    "n_two_three_wheeler": counts[2],
                }
            )
    return rows


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


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    parser = argparse.ArgumentParser(description="Build a YOLO subset from VisDrone-VID.")
    parser.add_argument(
        "--raw-dir", type=Path, default=None, help="defaults to settings.data.raw_dir"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="defaults to settings.data.subset_dir"
    )
    parser.add_argument(
        "--frame-stride", type=int, default=20, help="keep every Nth frame (all splits)"
    )
    parser.add_argument(
        "--sequences", nargs="+", default=None, help="restrict to these sequences (default: all)"
    )
    parser.add_argument("--link", action="store_true", help="symlink frames instead of copying")
    parser.add_argument(
        "--template-yaml", type=Path, default=None, help="defaults to settings.data.dataset_yaml"
    )
    args = parser.parse_args(argv)
    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")

    raw_dir = args.raw_dir or settings.data.raw_dir
    out_dir = args.out_dir or settings.data.subset_dir
    template = args.template_yaml or settings.data.dataset_yaml

    selected: dict[str, list[str]] = {
        split: list_sequences(raw_dir, split, only=args.sequences) for split in SPLITS
    }
    if args.sequences:
        matched = {seq for seqs in selected.values() for seq in seqs}
        if unknown := [s for s in args.sequences if s not in matched]:
            parser.error(f"--sequences not found in any split: {unknown}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "labels"):  # clean rebuild — drop any stale frames/labels
        if (out_dir / sub).exists():
            shutil.rmtree(out_dir / sub)

    rows: list[dict] = []
    for split in SPLITS:
        if not selected[split]:
            continue
        split_rows = build_split(
            raw_dir, out_dir, split, selected[split], args.frame_stride, args.link
        )
        rows.extend(split_rows)
        logger.info("%s: %d sequences -> %d frames", split, len(selected[split]), len(split_rows))

    write_manifest(rows, out_dir / "manifest.csv")
    yaml_path = stamp_dataset_yaml(template, out_dir)
    logger.info(
        "wrote %d images + manifest.csv + %s | boxes: person=%d vehicle=%d two-three-wheeler=%d",
        len(rows),
        yaml_path.name,
        sum(r["n_person"] for r in rows),
        sum(r["n_vehicle"] for r in rows),
        sum(r["n_two_three_wheeler"] for r in rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
