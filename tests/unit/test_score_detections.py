"""The detections scorer: ultralytics' validator over stored detections, on a 2-image dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("ultralytics")

from mlops_cv.evaluation.score_detections import score, score_with_ap_small  # noqa: E402

# Frame name -> (w, h) and its boxes as (cls, x1, y1, x2, y2) in source pixels.
FRAMES = {
    "a.jpg": ((320, 240), [(0, 10, 20, 60, 100), (1, 150, 50, 300, 200), (0, 200, 210, 220, 230)]),
    "b.jpg": ((200, 160), [(1, 40, 30, 120, 90)]),
}


@pytest.fixture
def data_yaml(tmp_path: Path) -> Path:
    import cv2

    for split_dir in ("images/test", "labels/test"):
        (tmp_path / split_dir).mkdir(parents=True)
    for name, ((w, h), boxes) in FRAMES.items():
        cv2.imwrite(str(tmp_path / "images/test" / name), np.zeros((h, w, 3), np.uint8))
        lines = [
            f"{c} {(x1 + x2) / 2 / w} {(y1 + y2) / 2 / h} {(x2 - x1) / w} {(y2 - y1) / h}"
            for c, x1, y1, x2, y2 in boxes
        ]
        (tmp_path / "labels/test" / Path(name).with_suffix(".txt")).write_text("\n".join(lines))
    path = tmp_path / "data.yaml"
    splits = dict.fromkeys(("train", "val", "test"), "images/test")
    names = {0: "person", 1: "vehicle"}
    path.write_text(yaml.safe_dump({"path": str(tmp_path), **splits, "names": names}))
    return path


def ground_truth() -> dict[str, np.ndarray]:
    return {
        name: np.array([[x1, y1, x2, y2, 1.0, c] for c, x1, y1, x2, y2 in boxes], np.float32)
        for name, (_, boxes) in FRAMES.items()
    }


def test_ground_truth_fed_back_scores_perfectly(data_yaml: Path) -> None:
    box = score(ground_truth(), data_yaml, "test").box
    assert (box.mp, box.mr) == (1.0, 1.0)
    assert box.map == pytest.approx(0.995)  # ultralytics' AP interpolation tops out at 0.995


def test_ap_small_comes_from_the_same_validator_pass(data_yaml: Path) -> None:
    """a.jpg's 20x20 person is the one small box; fed back, it scores perfectly."""
    metrics, ap_s = score_with_ap_small(ground_truth(), data_yaml, "test")
    assert metrics.box.map == pytest.approx(0.995)
    assert ap_s == pytest.approx(1.0)


def test_empty_predictions_score_zero(data_yaml: Path) -> None:
    empty = {name: np.zeros((0, 6), np.float32) for name in FRAMES}
    assert score(empty, data_yaml, "test").box.map == 0.0


def test_a_missing_frame_is_an_error_not_an_empty_prediction(data_yaml: Path) -> None:
    with pytest.raises(ValueError, match="b.jpg"):
        score({"a.jpg": ground_truth()["a.jpg"]}, data_yaml, "test")


def test_a_file_name_repeated_across_split_directories_is_an_error(data_yaml: Path) -> None:
    import shutil

    root = data_yaml.parent
    for kind in ("images", "labels"):
        shutil.copytree(root / kind / "test", root / kind / "test2")
    cfg = yaml.safe_load(data_yaml.read_text())
    data_yaml.write_text(yaml.safe_dump({**cfg, "test": ["images/test", "images/test2"]}))
    with pytest.raises(ValueError, match="repeat"):
        score(ground_truth(), data_yaml, "test")
