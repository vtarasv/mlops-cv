"""Unit tests for the VisDrone-VID -> YOLO converter (no dataset, no GPU, no PIL)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlops_cv.data.convert_visdrone_vid import (
    CLASS_MERGE,
    YoloBox,
    convert_annotation_line,
    convert_sequence,
)


def _line(
    frame: int = 1,
    tid: int = 0,
    left: float = 0,
    top: float = 0,
    w: float = 10,
    h: float = 10,
    score: int = 1,
    cat: int = 1,
    trunc: int = 0,
    occ: int = 0,
) -> str:
    """Build one VisDrone-VID annotation line."""
    return f"{frame},{tid},{left},{top},{w},{h},{score},{cat},{trunc},{occ}"


def test_class_merge_constant_exact() -> None:
    assert CLASS_MERGE == {1: 0, 2: 0, 4: 1, 5: 1, 6: 1, 9: 1, 3: 2, 7: 2, 8: 2, 10: 2}
    assert 0 not in CLASS_MERGE  # ignored regions
    assert 11 not in CLASS_MERGE  # others


@pytest.mark.parametrize(
    "raw,merged",
    [(1, 0), (2, 0), (4, 1), (5, 1), (6, 1), (9, 1), (3, 2), (7, 2), (8, 2), (10, 2)],
)
def test_each_raw_category_maps_to_expected_id(raw: int, merged: int) -> None:
    result = convert_annotation_line(_line(cat=raw, left=10, top=10, w=20, h=20), 100, 100)
    assert result is not None
    _, box = result
    assert box.cls == merged


def test_center_normalization() -> None:
    # left=100,top=50,w=40,h=20 in a 200x100 image -> center (0.6,0.6), size (0.2,0.2)
    result = convert_annotation_line(_line(left=100, top=50, w=40, h=20, cat=4), 200, 100)
    assert result is not None
    frame, box = result
    assert frame == 1
    assert box.cls == 1
    assert box.xc == pytest.approx(0.6)
    assert box.yc == pytest.approx(0.6)
    assert box.w == pytest.approx(0.2)
    assert box.h == pytest.approx(0.2)


def test_score_zero_skipped() -> None:
    assert convert_annotation_line(_line(score=0, cat=1), 100, 100) is None


@pytest.mark.parametrize("cat", [0, 11])
def test_ignored_and_others_categories_skipped(cat: int) -> None:
    assert convert_annotation_line(_line(score=1, cat=cat), 100, 100) is None


def test_partial_box_clipped_into_unit_range() -> None:
    # box straddles the top-left corner: corners clip to [0,0.2] in both axes
    result = convert_annotation_line(_line(left=-10, top=-10, w=30, h=30, cat=1), 100, 100)
    assert result is not None
    _, box = result
    assert box.xc == pytest.approx(0.1)
    assert box.yc == pytest.approx(0.1)
    assert box.w == pytest.approx(0.2)
    assert box.h == pytest.approx(0.2)
    assert 0.0 <= box.xc <= 1.0 and 0.0 <= box.yc <= 1.0


def test_box_fully_outside_frame_dropped() -> None:
    # left=200 on a 100px-wide image -> clips to zero width -> dropped
    assert convert_annotation_line(_line(left=200, top=10, w=20, h=20, cat=1), 100, 100) is None


@pytest.mark.parametrize("bad", ["", "   ", "1,2,3"])
def test_blank_or_malformed_lines_return_none(bad: str) -> None:
    assert convert_annotation_line(bad, 100, 100) is None


def test_convert_sequence_groups_by_frame_and_drops_all_skipped(tmp_path: Path) -> None:
    lines = [
        _line(frame=1, cat=1, left=10, top=10, w=20, h=20),  # person
        _line(frame=1, cat=4, left=30, top=30, w=20, h=20),  # vehicle
        _line(frame=2, score=0, cat=0, left=0, top=0, w=50, h=50),  # ignored region -> skip
        _line(frame=2, score=1, cat=11, left=5, top=5, w=10, h=10),  # others -> skip
        _line(frame=3, cat=10, left=40, top=40, w=20, h=20),  # two-three-wheeler
    ]
    ann = tmp_path / "seq.txt"
    ann.write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_frame = convert_sequence(ann, (100, 100))

    assert set(by_frame) == {1, 3}  # frame 2's rows are all skipped -> frame absent
    assert [b.cls for b in by_frame[1]] == [0, 1]
    assert [b.cls for b in by_frame[3]] == [2]


def test_yolobox_to_line_format() -> None:
    assert YoloBox(2, 0.5, 0.25, 0.1, 0.2).to_line() == "2 0.500000 0.250000 0.100000 0.200000"
