"""The FiftyOne loader's pure helpers (the script itself runs inside the App container)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from mlops_cv.autolabel.labeler import Labels, annotation_lines
from mlops_cv.data import convert_visdrone_vid

SCRIPT = Path(__file__).parents[2] / "docker" / "fiftyone" / "load_trial.py"
# The image copies the module beside the script; here it is the package's.
sys.modules.setdefault("convert_visdrone_vid", convert_visdrone_vid)
spec = importlib.util.spec_from_file_location("load_trial", SCRIPT)
assert spec and spec.loader
load_trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_trial)


def test_names() -> None:
    assert load_trial.dataset_name("VisDrone2019-VID-val") == "visdrone-vid-val"
    assert load_trial.field_name("detector-sam3-raw-train") == "detector_sam3_raw_train"


def test_auto_labels_carry_their_confidence_and_ground_truth_none() -> None:
    rows = np.array([[10, 20, 40, 30, 0.6, 1]], np.float32)
    text = "\n".join(annotation_lines(3, Labels(rows, ["van"], {}, 1, np.array([7]))))
    (row,) = load_trial.read_annotations(text)[3]
    (det,) = load_trial.boxes([row], (100, 50), scored=True)
    assert det["label"] == "vehicle" and det["raw_category"] == "van" and det["track_id"] == 7
    assert det["confidence"] == 0.6
    assert det["bounding_box"] == [0.1, 0.4, 0.3, 0.2]
    assert load_trial.boxes([row], (100, 50), scored=False)[0]["confidence"] is None


def test_ground_truth_splits_into_objects_and_ignored_regions() -> None:
    rows = ["1,0,0,0,40,50,0,0,0,0", "1,1,60,10,20,10,1,4,0,0", "1,3,70,30,10,10,0,11,0,0"]
    text = "\n".join([*rows, "2,1,1,1,1,1,1,1,0,0"])
    frames = load_trial.read_annotations(text)
    assert sorted(frames) == [1, 2]
    assert [d["label"] for d in load_trial.boxes(frames[1], (100, 50), scored=False)] == ["vehicle"]
    assert load_trial.ignored(frames[1], (100, 50)) == [
        {"label": "ignored", "bounding_box": [0.0, 0.0, 0.4, 1.0]}
    ]
