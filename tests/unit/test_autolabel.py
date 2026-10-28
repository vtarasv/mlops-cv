"""The auto-labeler's pure parts: VID reader, ignored-region rule, box filter, stage runner."""

from __future__ import annotations

import json
import typing
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from pydantic import ValidationError

from mlops_cv.autolabel import labeler, run, visdrone_vid
from mlops_cv.autolabel.labeler import BoxFilter, LabelingConfig, Labels
from mlops_cv.data.convert_visdrone_vid import YOLO_NAMES


def test_ignored_rule_uses_the_union_of_regions() -> None:
    # Each region covers 4 of the box's 10 columns: kept against either alone, dropped by both.
    box = np.array([[20, 1, 10, 10]])
    a, b = [21, 1, 3, 20], [26, 1, 3, 20]
    assert visdrone_vid.outside_ignored(box, np.array([a]), 64, 32).all()
    assert visdrone_vid.outside_ignored(box, np.array([b]), 64, 32).all()
    assert not visdrone_vid.outside_ignored(box, np.array([a, b]), 64, 32).any()


def test_ignored_rule_keeps_everything_without_regions() -> None:
    assert visdrone_vid.outside_ignored(np.array([[0, 0, 5, 5]]), np.zeros((0, 4)), 8, 8).all()


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    root = tmp_path / "raw" / "VisDrone2019-VID-val"
    seq = root / "sequences" / "uav1_v"
    seq.mkdir(parents=True)
    (root / "annotations").mkdir()
    for frame in range(1, 23):
        Image.new("RGB", (100, 50)).save(seq / f"{frame:07d}.jpg")
    rows = [
        "1,0,0,0,40,50,0,0,0,0",  # ignored region
        "1,1,60,10,20,10,1,4,0,0",  # car, kept
        "1,2,5,5,10,10,1,1,0,0",  # pedestrian inside the ignored region: dropped
        "1,3,70,30,10,10,0,11,0,0",  # others: never scored
        "2,1,61,10,20,10,1,4,0,0",  # frame 2 is not every-20th
        "21,1,62,10,20,10,1,4,0,0",
    ]
    (root / "annotations" / "uav1_v.txt").write_text("\n".join(rows) + "\n")
    return tmp_path / "raw"


def test_prepare_writes_the_scoring_set(raw_dir: Path, tmp_path: Path) -> None:
    first, last = visdrone_vid.prepare(raw_dir, tmp_path / "out", "val-stride20")
    out = visdrone_vid.scoring_dir(tmp_path / "out", "val-stride20")

    assert (first.name, last.name) == ("uav1_v_0000001.jpg", "uav1_v_0000021.jpg")
    assert first.categories.tolist() == [4] and last.boxes.tolist() == [[62, 10, 20, 10]]
    label = (out / "labels" / "uav1_v_0000001.txt").read_text()
    assert label == "1 0.700000 0.300000 0.200000 0.200000"
    link = out / "images" / "uav1_v_0000001.jpg"
    assert link.resolve() == raw_dir / "VisDrone2019-VID-val/sequences/uav1_v/0000001.jpg"
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert Path(data["path"]).is_absolute() and data["names"] == YOLO_NAMES
    gt = json.loads((out / "ground_truth.json").read_text())
    assert [a["attributes"]["raw_category"] for a in gt["annotations"]] == ["car", "car"]
    assert gt["annotations"][0]["category_id"] == 2
    ignored = json.loads((out / "ignored.json").read_text())
    assert [a["bbox"] for a in ignored["annotations"]] == [[0, 0, 40, 50]]


def test_frame_pickers() -> None:
    assert visdrone_vid.evenly_spaced(3)(list(range(1, 11))) == [1, 5, 10]
    assert visdrone_vid.evenly_spaced(10)([1, 2]) == [1, 2]
    assert visdrone_vid.strided(20)(list(range(1, 45))) == [1, 21, 41]


def test_the_champion_never_labels_train_frames() -> None:
    champion = LabelingConfig(detector={"model": labeler.CHAMPION})  # type: ignore[arg-type]
    assert not run.runnable(champion, "train-sample")
    assert run.runnable(champion, "val-stride20")


def test_merged_class_literal_matches_the_merge() -> None:
    assert set(typing.get_args(labeler.MergedClass)) == set(YOLO_NAMES.values())


def test_nested_rule_keeps_the_outer_box() -> None:
    boxes = np.array([[0, 0, 100, 100], [10, 10, 20, 20], [200, 200, 210, 210]], float)
    assert sorted(labeler.drop_nested(boxes, 0.8).tolist()) == [0, 2]


def test_filter_stages_and_top_k() -> None:
    rows = np.array(
        [
            [0, 0, 4, 40, 0.9, 0],  # side 4 < 8
            [0, 0, 90, 90, 0.9, 0],  # 81% of the frame
            [0, 0, 10, 10, 0.3, 0],  # score 0.3
            [20, 20, 30, 30, 0.5, 1],
            [40, 40, 50, 50, 0.7, 1],
        ],
        float,
    )
    f = BoxFilter(min_side=8, max_area=0.5, min_score=0.4, max_objects=1)
    assert labeler.filter_boxes(rows, f, 100, 100).tolist() == [4]
    assert labeler.filter_boxes(rows, BoxFilter(), 100, 100).tolist() == [0, 1, 4, 3, 2]


def test_filter_nms_is_class_agnostic() -> None:
    pytest.importorskip("torchvision")
    rows = np.array([[0, 0, 10, 10, 0.9, 0], [0, 0, 10, 11, 0.8, 1]], np.float32)
    assert labeler.filter_boxes(rows, BoxFilter(nms_iou=0.6), 100, 100).tolist() == [0]


def test_prompt_spans_split_at_separators() -> None:
    assert labeler.prompt_spans([101, 5, 1012, 7, 8, 1012, 102], {101, 102, 1012}) == [[1], [3, 4]]


def _stage(tmp_path: Path, stage: dict) -> Path:
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(stage))
    return path


def test_stage_trials_update_default_sections(tmp_path: Path) -> None:
    detector = {"model": "m", "prompts": {"car": "vehicle", "people": "person"}}
    stage = {
        "defaults": {"detector": detector, "box_filter": {"min_score": 0.4}},
        "trials": [
            {"name": "a", "detector": {"threshold": 0.1}},
            {"name": "b", "detector": {"prompts": {"vehicle": "vehicle"}}},
        ],
    }
    trials = run.load_stage(_stage(tmp_path, stage))
    a, b = trials["s-a"], trials["s-b"]
    assert a.detector.threshold == 0.1 and len(a.detector.prompts) == 2
    assert b.detector.prompts == {"vehicle": "vehicle"} and b.box_filter.min_score == 0.4


def test_stage_rejects_repeats_and_unknown_keys(tmp_path: Path) -> None:
    trial = {"name": "a", "detector": {"model": "m"}}
    with pytest.raises(ValueError, match="repeats"):
        run.load_stage(_stage(tmp_path, {"trials": [trial, trial]}))
    with pytest.raises(ValidationError):
        run.load_stage(_stage(tmp_path, {"trials": [{**trial, "box_filter": {"min_sid": 8}}]}))


def _image(name: str, categories: list[int]) -> visdrone_vid.Image:
    n = len(categories)
    return visdrone_vid.Image(
        name, Path(name), 10, 10, np.zeros((n, 4)), np.array(categories, int), np.zeros((0, 4))
    )


def test_cap_subset_and_recall_ceiling() -> None:
    images = [_image("a", [4] * 250 + [1] * 10), _image("b", [4] * 50)]
    assert run.uncapped(images, 200) == ["b"]
    assert run.recall_ceiling(images, 200) == pytest.approx(1 - 50 / 310)


def test_f1() -> None:
    assert run.f1(0.5, 1.0) == pytest.approx(2 / 3)
    assert run.f1(0.0, 0.0) == 0.0


def _labels(ms: float, masks: list | None = None, rows: np.ndarray | None = None) -> Labels:
    rows = np.zeros((0, 6)) if rows is None else rows
    return Labels(rows, ["car"] * len(rows), masks, {"detector": ms, "filter": 1.0}, len(rows))


def test_latency_leaves_out_warm_up() -> None:
    labels = [_labels(1000.0)] * run.WARMUP + [_labels(10.0), _labels(20.0)]
    out = run.latency(labels)
    assert out["latency/detector/p50_ms"] == 15.0
    assert out["latency/total/mean_ms"] == 16.0


def test_coco_autolabels_use_mask_boxes_and_skip_empty_masks() -> None:
    mask_utils = pytest.importorskip("pycocotools.mask")
    masks = np.zeros((20, 20, 2), np.uint8, order="F")
    masks[2:6, 3:9, 0] = 1  # the second mask stays empty
    rows = np.array([[0, 0, 10, 10, 0.9, 1], [5, 5, 8, 8, 0.8, 0]], np.float32)
    labels = _labels(1.0, mask_utils.encode(masks), rows)

    (ann,) = run.coco_autolabels([_image("a", [])], [labels])["annotations"]
    assert ann["bbox"] == [3, 2, 6, 4] and ann["det_bbox"] == [0, 0, 10, 10]
    assert (
        ann["area"] == 24
        and ann["category_id"] == 2
        and isinstance(ann["segmentation"]["counts"], str)
    )
    assert run.mask_rows(labels).tolist() == [[3, 2, 9, 6, pytest.approx(0.9), 1]]


def test_coco_autolabels_without_a_segmenter_keep_detector_boxes() -> None:
    pytest.importorskip("pycocotools")
    rows = np.array([[1, 2, 5, 8, 0.5, 0]], np.float32)
    (ann,) = run.coco_autolabels([_image("a", [])], [_labels(1.0, None, rows)])["annotations"]
    assert ann["bbox"] == ann["det_bbox"] == [1, 2, 4, 6] and "segmentation" not in ann
