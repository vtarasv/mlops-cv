"""Unit tests for the pure TP/FP matcher + segment selector."""

from __future__ import annotations

import pytest

from mlops_cv.evaluation.error_analysis import (
    Detection,
    GroundTruth,
    Matched,
    iou,
    match_detections,
    select_segments,
)


def test_iou_identical_is_one() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_is_zero() -> None:
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap_known_value() -> None:
    # Two unit squares offset 0.5 in both axes: inter 0.25, union 1.75 -> 1/7.
    assert iou((0, 0, 1, 1), (0.5, 0.5, 1.5, 1.5)) == pytest.approx(1.0 / 7.0)


def test_iou_zero_area_is_zero() -> None:
    assert iou((0, 0, 0, 10), (0, 0, 0, 10)) == 0.0


def _det(cls: int = 0, conf: float = 0.5, box: tuple = (0, 0, 10, 10)) -> Detection:
    return Detection(cls, conf, box)


def _gt(cls: int = 0, box: tuple = (0, 0, 10, 10)) -> GroundTruth:
    return GroundTruth(cls, box)


def test_match_single_tp() -> None:
    m = match_detections([_det()], [_gt()], 0.5)
    assert m[0].is_tp
    assert m[0].iou == pytest.approx(1.0)


def test_match_class_mismatch_is_fp() -> None:
    m = match_detections([_det(cls=0)], [_gt(cls=1)], 0.5)  # perfect overlap, wrong class
    assert not m[0].is_tp


def test_match_below_threshold_is_fp() -> None:
    m = match_detections([_det(box=(0, 0, 10, 10))], [_gt(box=(9, 9, 19, 19))], 0.5)
    assert not m[0].is_tp
    assert 0.0 < m[0].iou < 0.5  # they touch, but not enough to be a TP


def test_match_duplicate_predictions_one_tp_one_fp() -> None:
    # Two predictions on one GT: the higher-confidence one claims it (TP), the other is a dup (FP).
    m = match_detections([_det(conf=0.3), _det(conf=0.9)], [_gt()], 0.5)
    by_conf = {x.det.conf: x.is_tp for x in m}
    assert by_conf[0.9] is True
    assert by_conf[0.3] is False


def test_match_no_gt_all_fp() -> None:
    m = match_detections([_det(), _det(conf=0.9)], [], 0.5)
    assert all(not x.is_tp for x in m)


def _matched(conf: float, is_tp: bool, image: str = "a.jpg") -> Matched:
    return Matched(Detection(0, conf, (0, 0, 10, 10)), is_tp, 1.0 if is_tp else 0.0, image)


def test_select_low_conf_tp_ascending() -> None:
    low, high = select_segments([_matched(0.9, True), _matched(0.2, True), _matched(0.5, True)], 2)
    assert [m.det.conf for m in low] == [0.2, 0.5]
    assert high == []


def test_select_high_conf_fp_descending() -> None:
    pool = [_matched(0.3, False), _matched(0.95, False), _matched(0.6, False)]
    low, high = select_segments(pool, 2)
    assert [m.det.conf for m in high] == [0.95, 0.6]
    assert low == []


def test_select_fewer_than_k() -> None:
    low, high = select_segments([_matched(0.5, True)], 10)
    assert len(low) == 1 and high == []


def test_select_deterministic_tie_break_by_image() -> None:
    pool = [_matched(0.5, False, "b.jpg"), _matched(0.5, False, "a.jpg")]
    _, high = select_segments(pool, 2)
    assert [m.image for m in high] == ["a.jpg", "b.jpg"]  # equal conf -> image ascending
