"""Tests for the subset layout module: path constructors, manifest IO, demo-store probe."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlops_cv.data.subset import (
    MANIFEST_FIELDS,
    SPLITS,
    csv_line,
    demo_image_relpath,
    demo_label_relpath,
    demo_store_present,
    frame_name,
    image_relpath,
    label_relpath,
    manifest_rows,
    raw_split_dir,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "split": "train",
        "sequence": "uav0000001_00000_v",
        "frame_index": 7,
        "image_relpath": image_relpath("train", "uav0000001_00000_v_0000007"),
        "label_relpath": label_relpath("train", "uav0000001_00000_v_0000007"),
        "width": 1920,
        "height": 1080,
        "n_boxes": 3,
        "n_person": 1,
        "n_vehicle": 2,
        "n_two_three_wheeler": 0,
    }
    row.update(overrides)
    return row


def test_frame_name_pads_to_seven_digits() -> None:
    assert frame_name("seqA", 7) == "seqA_0000007"
    assert frame_name("seqA", 1234567) == "seqA_1234567"


def test_relpaths_are_posix_style_and_split_scoped() -> None:
    assert image_relpath("val", "seqA_0000007") == "images/val/seqA_0000007.jpg"
    assert label_relpath("val", "seqA_0000007") == "labels/val/seqA_0000007.txt"
    assert demo_image_relpath("seqB", 3) == "demo/images/seqB/0000003.jpg"
    assert demo_label_relpath("seqB", 3) == "demo/labels/seqB/0000003.txt"


def test_raw_split_dir_maps_yolo_names() -> None:
    assert [raw_split_dir(s) for s in SPLITS] == [
        "VisDrone2019-VID-train",
        "VisDrone2019-VID-val",
        "VisDrone2019-VID-test-dev",
    ]
    with pytest.raises(KeyError):
        raw_split_dir("test-dev")  # raw suffixes are internal, not an interface vocabulary


def test_manifest_write_read_round_trip() -> None:
    rows = [_row(), _row(split="val", frame_index=9, n_boxes=0, n_vehicle=0, n_person=0)]
    text = "\n".join([",".join(MANIFEST_FIELDS), *(csv_line(r, MANIFEST_FIELDS) for r in rows)])
    parsed = manifest_rows(text.splitlines())
    assert [list(r.keys()) for r in parsed] == [MANIFEST_FIELDS, MANIFEST_FIELDS]
    assert [r["frame_index"] for r in parsed] == ["7", "9"]  # CSV strings by design
    assert parsed[0]["image_relpath"] == "images/train/uav0000001_00000_v_0000007.jpg"


def test_manifest_rows_rejects_wrong_header() -> None:
    with pytest.raises(ValueError, match="unexpected manifest header"):
        manifest_rows(["split,sequence", "train,seqA"])


def test_demo_store_present(tmp_path: Path) -> None:
    assert not demo_store_present(tmp_path)  # no demo dir at all
    frame = tmp_path / demo_image_relpath("seqA", 1)
    frame.parent.mkdir(parents=True)
    assert not demo_store_present(tmp_path)  # empty sequence dir
    frame.write_bytes(b"\xff\xd8\xff")
    assert demo_store_present(tmp_path)
