"""Golden: the names and the dump format the edge bench and the desktop gate share."""

from __future__ import annotations

import json

import numpy as np

from mlops_cv.tracking import edge_record

# The dump as it lands on the record run: frame file name -> rows of x1 y1 x2 y2 conf cls.
GOLDEN_DUMP = (
    '{"uav0000009_03358_v_0000001.jpg": [[10.5, 20.25, 30.0, 40.75, 0.875, 1.0]], '
    '"uav0000009_03358_v_0000012.jpg": []}'
)


def test_names_are_pinned() -> None:
    assert edge_record.detections_path("pi5", "ncnn-fp16-640") == (
        "edge/pi5/ncnn-fp16-640/detections.json"
    )
    assert edge_record.report_path("pi5") == "edge/pi5.md"
    assert edge_record.package_tag("pi5", "ncnn") == "optimize.pi5.ncnn"
    assert edge_record.measured_tag("pi5") == "optimize.measured.pi5"


def test_dump_is_pinned_and_round_trips() -> None:
    dets = {
        "uav0000009_03358_v_0000001.jpg": np.array([[10.5, 20.25, 30, 40.75, 0.875, 1]]),
        "uav0000009_03358_v_0000012.jpg": np.zeros((0, 6), np.float32),
    }
    assert json.dumps(edge_record.to_payload(dets)) == GOLDEN_DUMP
    back = edge_record.from_payload(json.loads(GOLDEN_DUMP))
    assert back["uav0000009_03358_v_0000012.jpg"].shape == (0, 6)
    for name, rows in dets.items():
        np.testing.assert_array_equal(back[name], rows.astype(np.float32))


def test_package_versions_read_back_only_this_device() -> None:
    tags = {
        "optimize.pi5.ncnn": "1.0",
        "optimize.pi5.numpy": "2.4.6",
        "optimize.jetson.ncnn": "0.9",
        "optimize.measured.pi5": "true",
    }
    assert edge_record.package_versions(tags, "pi5") == {"ncnn": "1.0", "numpy": "2.4.6"}
