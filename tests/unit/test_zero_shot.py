"""Unit tests for the zero-shot COCO -> merged class mapping (no GPU, no ultralytics)."""

from __future__ import annotations

import numpy as np

from mlops_cv.evaluation.zero_shot import class_lut, merge_classes

COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 9: "bird"}


def test_lut_maps_coco_onto_merged_and_drops_the_rest() -> None:
    lut = class_lut(COCO)
    assert lut.tolist() == [0, 2, 1, 2, -1, 1, -1, 1, -1, -1]


def test_merge_drops_unmapped_and_keeps_one_copy_of_a_same_box_duplicate() -> None:
    rows = np.array(
        [
            [0, 0, 10, 10, 0.2, 7],  # truck: same box as the car, same merged class -> dropped
            [0, 0, 10, 10, 0.6, 2],  # car
            [0, 0, 10, 10, 0.3, 0],  # person on the same box: a different merged class, kept
            [5, 5, 9, 9, 0.9, 9],  # bird: dropped
            [1, 1, 4, 4, 0.1, 3],  # motorcycle
        ]
    )
    merged = merge_classes(rows, class_lut(COCO))
    assert merged.tolist() == [
        [0, 0, 10, 10, 0.6, 1],
        [0, 0, 10, 10, 0.3, 0],
        [1, 1, 4, 4, 0.1, 2],
    ]
