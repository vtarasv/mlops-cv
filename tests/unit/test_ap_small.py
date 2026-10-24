"""mAP50-95_small: the YOLO-labels -> COCO ground-truth bridge, scored by faster-coco-eval."""

from __future__ import annotations

import numpy as np
import pytest

from mlops_cv.evaluation.ap_small import coco_ground_truth

NAMES = {0: "person", 1: "vehicle"}
# One 200x100 frame: a 10x10 person (small) and a 100x50 vehicle (large), normalized xywh.
LABELS = [
    {
        "im_file": "/subset/images/val/0001.jpg",
        "shape": (100, 200),
        "cls": np.array([[0.0], [1.0]]),
        "bboxes": np.array([[0.5, 0.5, 0.05, 0.1], [0.25, 0.25, 0.5, 0.5]]),
    }
]


def _pred(category_id: int, bbox: list[float], score: float = 0.9) -> dict:
    # pred_to_json's shape: a numeric stem becomes an int image_id ("0001" -> 1).
    return {"image_id": 1, "file_name": "0001.jpg", "category_id": category_id, "bbox": bbox,
            "score": score}  # fmt: skip


SMALL = _pred(1, [95.0, 45.0, 10.0, 10.0])
LARGE = _pred(2, [0.0, 0.0, 100.0, 50.0])


def test_ground_truth_is_source_pixel_xywh_with_shifted_category_ids() -> None:
    gt, ids = coco_ground_truth(LABELS, NAMES)
    assert ids == {"0001": 1}
    assert gt["images"] == [{"id": 1, "width": 200, "height": 100, "file_name": "0001"}]
    small, large = gt["annotations"]
    assert (small["category_id"], small["bbox"], small["area"]) == (
        1,
        [95.0, 45.0, 10.0, 10.0],
        100.0,
    )
    assert (large["category_id"], large["bbox"]) == (2, [0.0, 0.0, 100.0, 50.0])
    assert [c["id"] for c in gt["categories"]] == [1, 2]


def test_a_repeated_stem_is_an_error() -> None:
    with pytest.raises(ValueError, match="repeats"):
        coco_ground_truth(LABELS * 2, NAMES)


def test_ap_small_scores_only_the_small_bucket() -> None:
    pytest.importorskip("faster_coco_eval")
    from mlops_cv.evaluation.ap_small import ap_small, coco_stats

    assert ap_small([SMALL, LARGE], LABELS, NAMES) == pytest.approx(1.0)
    assert ap_small([LARGE], LABELS, NAMES) == 0.0
    assert coco_stats([LARGE], LABELS, NAMES)["AP_all"] == pytest.approx(0.5)
    assert ap_small([], LABELS, NAMES) == 0.0
