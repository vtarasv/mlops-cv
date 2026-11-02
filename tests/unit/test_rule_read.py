"""The VisDrone-rule read scores the same detections under both rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("ultralytics")

from mlops_cv.tuning.rule_read import both_rules  # noqa: E402
from mlops_cv.tuning.visdrone_vid import Image  # noqa: E402


def test_a_false_positive_in_an_ignored_region_counts_only_under_the_repo_rule(
    tmp_path: Path,
) -> None:
    import cv2

    for kind in ("images", "labels"):
        (tmp_path / kind / "val").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "images/val/a.jpg"), np.zeros((100, 200, 3), np.uint8))
    (tmp_path / "labels/val/a.txt").write_text("0 0.75 0.5 0.2 0.4\n")  # x1 y1 x2 y2 130 30 170 70
    data_yaml = tmp_path / "data.yaml"
    splits = dict.fromkeys(("train", "val"), "images/val")
    data_yaml.write_text(yaml.safe_dump({"path": str(tmp_path), **splits, "names": {0: "person"}}))
    none = np.zeros(0)
    image = Image(
        name="a.jpg",
        path=Path("a.jpg"),
        width=200,
        height=100,
        boxes=np.zeros((0, 4)),
        categories=none,
        track_ids=none,
        ignored=np.array([[0, 0, 100, 100]]),
        sequence="s",
        frame=1,
    )
    rows = np.array([[130, 30, 170, 70, 0.9, 0], [10, 10, 50, 50, 0.95, 0]], np.float32)
    out = both_rules({"a.jpg": rows}, [image], data_yaml, data_yaml)
    assert out["val-visdrone-rule/precision"] == 1.0
    assert out["val/precision"] < 1.0
