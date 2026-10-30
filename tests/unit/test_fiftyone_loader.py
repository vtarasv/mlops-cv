"""The FiftyOne loader's pure helpers (the script itself runs inside the App container)."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import numpy as np

from mlops_cv.autolabel import run, view

SCRIPT = Path(__file__).parents[2] / "docker-compose" / "fiftyone" / "load_trial.py"
spec = importlib.util.spec_from_file_location("load_trial", SCRIPT)
assert spec and spec.loader
load_trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_trial)


def test_trial_file_path_names_its_set_and_field_prefix(tmp_path: Path) -> None:
    path = view.trial_path(tmp_path, "val-stride20", "smoke-gdino-t-sam2.1-s-val-stride20")

    assert path == tmp_path / "val-stride20" / "trials" / "smoke-gdino-t-sam2.1-s-val-stride20.json"
    assert load_trial.dataset_name(path.parent.parent.name) == "visdrone-vid-val-stride20"
    assert load_trial.field_prefix(path.stem, "val-stride20") == "smoke_gdino_t_sam2_1_s"
    assert run.autolabels_file("val-stride20") == "autolabels-val-stride20.json"


def test_frame_fields_split_the_flattened_name() -> None:
    assert load_trial.frame_fields("uav0000013_00000_v_0000021.jpg") == {
        "sequence": "uav0000013_00000_v",
        "frame_number": 21,
    }


def test_detector_boxes_drop_the_mask_and_keep_the_prompt() -> None:
    ann = {
        "bbox": [1, 1, 2, 2],
        "det_bbox": [0, 0, 4, 4],
        "segmentation": {"size": [8, 8], "counts": "x"},
        "attributes": {"label": "car"},
    }

    tracked = {"bbox": [1, 1, 2, 2], "track_id": 3}  # a tracked frame has no detector box
    (det,) = load_trial.detector_boxes([load_trial.with_prompt(ann), tracked])

    assert det == {"bbox": [0, 0, 4, 4], "attributes": {"prompt": "car"}}
    assert ann["attributes"] == {"label": "car"}  # the input is left alone


def test_mask_crops_are_copied_so_the_frame_is_freed() -> None:
    frame = np.ones((8, 8), bool)
    fouc = types.SimpleNamespace(_coco_segmentation_to_mask=lambda seg, *_: seg and frame[1:3])
    load_trial.copy_cropped_masks(fouc)
    crop = fouc._coco_segmentation_to_mask({"counts": "x"}, [0, 0, 1, 1], (8, 8))
    assert crop.shape == (2, 8) and crop.base is None
    assert fouc._coco_segmentation_to_mask(None) is None
