"""The VisDrone-rule read scores the same detections under both rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("ultralytics")

from mlops_cv.tuning.autolabel_visdrone_vid.rule_read import both_rules  # noqa: E402
from mlops_cv.tuning.autolabel_visdrone_vid.visdrone_vid import Image, readings  # noqa: E402

HIT = np.array([[130, 30, 170, 70, 0.6, 0]], np.float32)  # a.jpg's one ground-truth box, at 0.6


@pytest.fixture
def data_yaml(tmp_path: Path) -> Path:
    import cv2

    for kind in ("images", "labels"):
        (tmp_path / kind / "val").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "images/val/a.jpg"), np.zeros((100, 200, 3), np.uint8))
    (tmp_path / "labels/val/a.txt").write_text("0 0.75 0.5 0.2 0.4\n")  # x1 y1 x2 y2 130 30 170 70
    path = tmp_path / "data.yaml"
    splits = dict.fromkeys(("train", "val"), "images/val")
    path.write_text(yaml.safe_dump({"path": str(tmp_path), **splits, "names": {0: "person"}}))
    return path


def test_a_false_positive_in_an_ignored_region_counts_only_under_the_repo_rule(
    data_yaml: Path,
) -> None:
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


def test_readings_carry_the_max_f1_confidence(data_yaml: Path) -> None:
    """A hit at 0.6 and a miss at 0.3: F1 peaks between them, where the miss is cut."""
    miss = np.array([[0, 0, 5, 5, 0.3, 0]], np.float32)
    dets = {"a.jpg": np.concatenate([HIT, miss])}
    assert 0.3 < readings(dets, data_yaml, "val", "x")["x/conf_at_max_F1"] < 0.6


def test_readings_at_a_floor_read_the_curves_at_that_confidence(data_yaml: Path) -> None:
    """The hit scores 0.6: a floor below it keeps it, a floor above it doesn't."""
    dets = {"a.jpg": HIT}
    assert readings(dets, data_yaml, "val", "x", floor=0.55)["x/at_floor/recall"] == 1.0
    assert readings(dets, data_yaml, "val", "x", floor=0.65)["x/at_floor/recall"] == 0.0
    assert "x/at_floor/recall" not in readings(dets, data_yaml, "val", "x")
