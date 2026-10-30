"""The auto-labeler's pure parts: VID reader, ignored-region rule, box filter, stage runner,
key-frame tracking rules and the TrackEval glue."""

from __future__ import annotations

import json
import typing
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from pydantic import ValidationError

from mlops_cv.autolabel import labeler, run, sequence, track_scores, visdrone_vid
from mlops_cv.autolabel.labeler import BoxFilter, LabelingConfig, Labels, Tracker
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
    assert [a["track_id"] for a in gt["annotations"]] == [1, 1]
    assert gt["annotations"][0]["category_id"] == 2
    ignored = json.loads((out / "ignored.json").read_text())
    assert [a["bbox"] for a in ignored["annotations"]] == [[0, 0, 40, 50]]


def test_full_rate_set_reads_every_frame_with_track_ids(raw_dir: Path, tmp_path: Path) -> None:
    images = visdrone_vid.prepare(raw_dir, tmp_path / "out", "val-full")

    assert [im.frame for im in images] == list(range(1, 23))
    assert {im.sequence for im in images} == {"uav1_v"}
    first = images[0]  # the pedestrian inside the ignored region is dropped with its track id
    assert first.track_ids.tolist() == [1] and first.categories.tolist() == [4]


def test_frame_pickers() -> None:
    assert visdrone_vid.evenly_spaced(3)(list(range(1, 11))) == [1, 5, 10]
    assert visdrone_vid.evenly_spaced(10)([1, 2]) == [1, 2]
    assert visdrone_vid.strided(20)(list(range(1, 45))) == [1, 21, 41]


def _config(**sections: object) -> LabelingConfig:
    return LabelingConfig.model_validate({"detector": {"model": "m"}, **sections})


def test_skip_reasons() -> None:
    champion = _config(detector={"model": labeler.CHAMPION})
    assert run.skip_reason(champion, "train-sample")
    assert run.skip_reason(champion, "val-stride20") is None
    tracked = _config(tracker={"model": "botsort"})
    assert run.skip_reason(tracked, "val-stride20")
    assert run.skip_reason(tracked, "val-full") is None


def test_a_video_tracker_makes_its_own_masks() -> None:
    with pytest.raises(ValidationError, match="own masks"):
        _config(segmenter="s", tracker={"model": "edgetam"})
    _config(segmenter="s", tracker={"model": "botsort"})


def test_merged_class_literal_matches_the_merge() -> None:
    assert set(typing.get_args(labeler.MergedClass)) == set(YOLO_NAMES.values())


def test_nested_rule_keeps_the_outer_or_the_inner_box() -> None:
    boxes = np.array([[0, 0, 100, 100], [10, 10, 20, 20], [200, 200, 210, 210]], float)
    assert sorted(labeler.drop_nested(boxes, 0.8).tolist()) == [0, 2]
    assert sorted(labeler.drop_nested(boxes, 0.8, keep_outer=False).tolist()) == [1, 2]


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


def test_slice_grid_matches_sahi() -> None:
    grid = labeler.slice_grid(1360, 765, 640)
    assert sorted({x for x, *_ in grid}) == [0, 512, 720] and sorted({y for _, y, *_ in grid}) == [
        0,
        125,
    ]
    assert all(x1 - x0 == 640 and y1 - y0 == 640 for x0, y0, x1, y1 in grid)
    assert labeler.slice_grid(500, 300, 640) == [(0, 0, 500, 300)]


def test_tiled_shifts_each_pass_and_adds_the_full_frame() -> None:
    def detect(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        return np.array([[1, 2, 3, 4, 0.5, 1]], np.float32), ["car"]

    image = np.zeros((10, 16, 3), np.uint8)
    rows, labels, windows = labeler.tiled(detect, image, 10)
    assert windows.tolist() == [[0, 0, 10, 10], [6, 0, 16, 10], [0, 0, 16, 10]]
    assert rows[:, :2].tolist() == [[1, 2], [7, 2], [1, 2]] and labels == ["car"] * 3
    assert labeler.tiled(detect, image, 20)[2].tolist() == [[0, 0, 16, 10]]


def test_edge_rule_drops_boxes_cut_by_interior_slice_edges_only() -> None:
    boxes = np.array([[50, 10, 59, 20], [0, 0, 20, 20], [50, 10, 59, 20]], np.float32)
    windows = np.array([[0, 0, 60, 60], [0, 0, 60, 60], [0, 0, 100, 100]], np.float32)
    assert labeler.cut_by_slice_edge(boxes, windows, 100, 100).tolist() == [True, False, False]


def test_merge_passes_with_the_edge_rule_then_class_aware_nms() -> None:
    pytest.importorskip("torchvision")
    rows = np.array(
        [
            [40, 10, 59, 20, 0.9, 1],  # cut by the slice's right edge
            [40, 10, 70, 20, 0.6, 1],  # the whole object, from the full frame
            [40, 10, 70, 21, 0.5, 1],  # its duplicate
            [40, 10, 70, 20, 0.4, 0],  # another class
        ],
        np.float32,
    )
    windows = np.array([[0, 0, 60, 60]] + [[0, 0, 100, 100]] * 3, np.float32)
    keep, conf = labeler.merge_passes(rows, windows, "edge", 100, 100)
    assert keep.tolist() == [1, 3] and conf.tolist() == pytest.approx([0.6, 0.4])


def test_contained_merge_keeps_the_larger_box_with_the_pair_top_score() -> None:
    pytest.importorskip("torchvision")
    rows = np.array(
        [[40, 10, 52, 20, 0.9, 1], [40, 10, 70, 20, 0.6, 1], [40, 10, 52, 20, 0.8, 0]], np.float32
    )
    windows = np.array([[0, 0, 60, 60]] * 3, np.float32)
    keep, conf = labeler.merge_passes(rows, windows, "contained", 100, 100)
    assert keep.tolist() == [1, 2] and conf.tolist() == pytest.approx([0.9, 0.8])


def test_prompt_spans_split_at_separators() -> None:
    assert labeler.prompt_spans([101, 5, 1012, 7, 8, 1012, 102], {101, 102, 1012}) == [[1], [3, 4]]


def test_top_rows_keep_the_best_boxes_above_the_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(labeler, "MAX_DET", 2)
    cfg = labeler.Detector(model="m", prompts={"car": "vehicle", "person": "person"}, threshold=0.1)
    boxes = torch.arange(4.0)[:, None] + torch.tensor([0.0, 0, 1, 1])
    conf = torch.tensor([0.05, 0.9, 0.5, 0.7])
    rows, labels = labeler.top_rows(boxes, conf, torch.tensor([0, 1, 0, 0]), cfg)
    assert rows[:, 0].tolist() == [1, 3] and labels == ["person", "car"]
    assert rows[:, 5].tolist() == [labeler.MERGED["person"], labeler.MERGED["vehicle"]]
    empty, _ = labeler.top_rows(boxes, conf * 0, torch.tensor([0, 1, 0, 0]), cfg)
    assert empty.shape == (0, 6)


def test_detector_loader_is_picked_by_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("grounding_dino", "owlv2", "sam3", "yoloe"):
        monkeypatch.setattr(labeler, name, lambda cfg, device, name=name: name)
    pick = {
        "IDEA-Research/grounding-dino-base": "grounding_dino",
        "google/owlv2-base-patch16-ensemble": "owlv2",
        "facebook/sam3": "sam3",
        "yoloe-26l-seg.pt": "yoloe",
    }
    for model, loader in pick.items():
        assert labeler.detector(labeler.Detector(model=model), "cpu") == loader


def test_an_unset_detector_field_keeps_the_stored_detections_key() -> None:
    import hashlib

    before_imgsz = '{"model":"m","prompts":{},"threshold":0.001}'
    key = hashlib.sha256(before_imgsz.encode()).hexdigest()[:16]
    assert run.detections_store(Path("s"), _config()) == Path(f"s/detections/{key}.npz")
    sized = _config(detector={"model": "m", "imgsz": 1280})
    assert run.detections_store(Path("s"), sized) != run.detections_store(Path("s"), _config())
    bf16 = _config(detector={"model": "m", "dtype": "bfloat16"})
    assert run.detections_store(Path("s"), bf16) != run.detections_store(Path("s"), _config())
    edge, contained, other = (
        _config(detector={"model": "m", "tiling": {"side": side, "merge": merge}})
        for side, merge in [(806, "edge"), (806, "contained"), (504, "edge")]
    )
    assert run.detections_store(Path("s"), edge) == run.detections_store(Path("s"), contained)
    assert run.detections_store(Path("s"), edge) != run.detections_store(Path("s"), other)


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


def _image(
    name: str,
    categories: list[int],
    boxes: list | None = None,
    track_ids: list[int] | None = None,
    sequence: str = "s",
) -> visdrone_vid.Image:
    n = len(categories)
    return visdrone_vid.Image(
        name,
        Path(name),
        100,
        100,
        np.zeros((n, 4)) if boxes is None else np.array(boxes, float),
        np.array(categories, int),
        np.arange(n) if track_ids is None else np.array(track_ids),
        np.zeros((0, 4)),
        sequence,
        1,
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


def test_coco_autolabels_leave_out_scores_below_the_floor() -> None:
    pytest.importorskip("pycocotools")
    rows = np.array([[1, 2, 5, 8, 0.5, 0], [1, 2, 5, 8, 0.2, 0]], np.float32)
    coco = run.coco_autolabels([_image("a", [])], [_labels(1.0, None, rows)], floor=0.3)
    assert [a["score"] for a in coco["annotations"]] == [0.5]


def test_stored_detections_round_trip_and_check_their_frames(tmp_path: Path) -> None:
    images = [_image("a", []), _image("b", [])]
    rows = np.array([[1, 2, 5, 8, 0.5, 0]], np.float32)
    raw = [Labels(rows, ["car"], None, {"detector": 7.0}, 1), _labels(3.0)]
    path = tmp_path / "d.npz"
    run.save_detections(path, images, raw)

    a, b = run.load_detections(path, images)
    assert a.rows.tolist() == rows.tolist() and a.labels == ["car"] and a.ms == {"detector": 7.0}
    assert (len(b.rows), b.detected, b.ms) == (0, 0, {"detector": 3.0})
    with pytest.raises(ValueError, match="other frames"):
        run.load_detections(path, images[:1])
    assert a.windows is None


def test_stored_tiled_detections_keep_their_windows(tmp_path: Path) -> None:
    rows = np.array([[1, 2, 5, 8, 0.5, 0]], np.float32)
    windows = np.array([[0, 0, 10, 10]], np.float32)
    raw = [Labels(rows, ["car"], None, {"detector": 7.0}, 1, windows=windows)]
    raw.append(
        Labels(
            np.zeros((0, 6), np.float32),
            [],
            None,
            {"detector": 1.0},
            0,
            windows=np.zeros((0, 4), np.float32),
        )
    )
    path = tmp_path / "d.npz"
    run.save_detections(path, [_image("a", []), _image("b", [])], raw)
    a, b = run.load_detections(path, [_image("a", []), _image("b", [])])
    assert a.windows is not None and a.windows.tolist() == windows.tolist()
    assert b.windows is not None and b.windows.shape == (0, 4)


def test_coco_autolabels_carry_track_ids_and_skip_missing_detector_boxes() -> None:
    pytest.importorskip("pycocotools")
    rows = np.array([[1, 2, 5, 8, 0.5, 0], [np.nan, np.nan, np.nan, np.nan, 0.7, 1]], np.float32)
    labels = Labels(rows, ["a", "b"], None, {}, 1, np.array([3, 4]))
    first, second = run.coco_autolabels([_image("a", [])], [labels])["annotations"]
    assert (first["track_id"], second["track_id"]) == (3, 4)
    assert "det_bbox" in first and "det_bbox" not in second


def test_amortised_latency_drops_whole_key_frame_cycles() -> None:
    key = Labels(np.zeros((0, 6)), [], None, {"detector": 100.0, "tracker": 10.0}, 0)
    tracked = Labels(np.zeros((0, 6)), [], None, {"tracker": 10.0}, 0)
    labels = ([key] + [tracked] * 4) * 3  # K = 5; the first cycle is the warm-up
    out = run.amortised_latency(labels, Tracker(model="m", key_interval=5))
    assert out["latency/detector/amortised_ms"] == 20.0
    assert out["latency/total/amortised_ms"] == 30.0


def test_match_assigns_one_to_one_and_every_unmatched_box_is_new() -> None:
    pytest.importorskip("lap")
    iou = np.array([[0.9, 0.0], [0.6, 0.1], [0.0, 0.4], [0.1, 0.2]])
    pairs, new = sequence.match(iou, 0.3)
    assert pairs == {0: 0, 2: 1}
    assert new.tolist() == [1, 3]  # box 1 lost track 0 to box 0, so it starts a track


def test_match_maximises_the_total_iou_not_the_best_pair() -> None:
    pytest.importorskip("lap")
    pairs, _ = sequence.match(np.array([[0.9, 0.8], [0.85, 0.0]]), 0.3)
    assert pairs == {0: 1, 1: 0}  # greedy would take 0.9 and leave box 1 without its track


def test_match_without_tracks_makes_every_box_new() -> None:
    pytest.importorskip("lap")
    pairs, new = sequence.match(np.zeros((2, 0)), 0.3)
    assert pairs == {} and new.tolist() == [0, 1]


def test_the_younger_duplicate_ends() -> None:
    iou = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.9], [0.0, 0.9, 1.0]])
    assert sequence.younger_duplicates(iou, [4, 2, 7], 0.8) == {4, 7}


def _rows(*scores: float) -> np.ndarray:
    return np.array([[0, 0, 1, 1, s, 1] for s in scores], float).reshape(-1, 6)


def test_key_frames_start_match_and_end_tracks() -> None:
    tracks = sequence.Tracks(Tracker(model="m", unmatched_keys=2))
    ended, boxes = tracks.key_frame([], _rows(0.9, 0.5), ["a", "b"], np.zeros((2, 0)), np.eye(0))
    assert ended == set() and boxes == {1: 0, 2: 1}
    assert tracks.live[2].label == "b" and tracks.live[2].cls == 1

    # Track 1 matches a fresher box, track 2 misses its first key frame.
    iou = np.array([[0.8, 0.0]])
    ended, boxes = tracks.key_frame([1, 2], _rows(0.6), ["a"], iou, np.eye(2))
    assert ended == set() and boxes == {1: 0} and tracks.live[1].score == 0.6
    ended, _ = tracks.key_frame([1, 2], _rows(0.6), ["a"], iou, np.eye(2))
    assert ended == {2} and list(tracks.live) == [1]


def test_duplicates_end_before_matching_and_the_cap_keeps_the_best_new_boxes() -> None:
    tracks = sequence.Tracks(Tracker(model="m", max_objects=3))
    tracks.key_frame([], _rows(0.9, 0.8), ["a", "a"], np.zeros((2, 0)), np.eye(0))
    mask_iou = np.array([[1.0, 0.95], [0.95, 1.0]])
    box_iou = np.array([[0.1, 0.9], [0.0, 0.0], [0.0, 0.0]])  # box 0 fits track 2 only
    ended, boxes = tracks.key_frame([1, 2], _rows(0.9, 0.7, 0.6), ["a"] * 3, box_iou, mask_iou)
    assert ended == {2}
    assert boxes == {3: 0, 4: 1}  # box 0 is new, as its only match ended; the cap drops box 2
    assert tracks.live[1].missed == 1


def test_empty_masks_end_a_track() -> None:
    tracks = sequence.Tracks(Tracker(model="m", empty_frames=2))
    tracks.key_frame([], _rows(0.9), ["a"], np.zeros((1, 0)), np.eye(0))
    assert tracks.observe([1], [0]) == set()
    assert tracks.observe([1], [5]) == set()  # a non-empty frame resets the count
    assert tracks.observe([1], [0]) == set()
    assert tracks.observe([1], [0]) == {1} and not tracks.live


class _Session:
    output_dict_per_obj: dict[int, typing.Any]

    def __init__(self, ids: list[int]) -> None:
        self._obj_id_to_idx = {o: i for i, o in enumerate(ids)}
        self._obj_idx_to_id = dict(enumerate(ids))
        self.obj_ids = list(ids)
        self.obj_with_new_inputs: list[int] = []
        self.processed_frames = {30: "frame"}
        for store in sequence.PER_OBJECT:
            setattr(self, store, {i: f"{store}-{o}" for i, o in enumerate(ids)})


def test_set_objects_reindexes_the_session() -> None:
    session = _Session([5, 6, 7])
    state = sequence.objects(session)
    sequence.set_objects(session, {o: s for o, s in state.items() if o != 6})
    assert session.obj_ids == [5, 7] and session._obj_idx_to_id == {0: 5, 1: 7}
    assert session.output_dict_per_obj == {
        0: "output_dict_per_obj-5",
        1: "output_dict_per_obj-7",
    }
    sequence.set_objects(session, {})
    sequence.set_objects(session, state)
    assert sequence.objects(session) == state


def test_prune_keeps_seed_frames_and_the_last_window() -> None:
    session = _Session([1])
    frames = {t: {"high_res_masks": 1} for t in range(31)}
    out = {"cond_frame_outputs": {0: {"high_res_masks": 1}}, "non_cond_frame_outputs": frames}
    session.output_dict_per_obj = {0: out}
    sequence.prune(session, 30)
    assert sorted(out["non_cond_frame_outputs"]) == list(range(30 - sequence.KEEP_FRAMES, 31))
    assert "high_res_masks" not in out["non_cond_frame_outputs"][30]
    assert list(out["cond_frame_outputs"]) == [0] and session.processed_frames == {}


def test_mot_files_split_by_class_and_number_frames_per_sequence() -> None:
    images = [
        _image("a1", [1, 4], [[0, 0, 10, 10], [50, 50, 10, 10]], [7, 8], "a"),
        _image("b1", [4], [[5, 5, 10, 10]], [9], "b"),
    ]
    tracked = [np.array([[0, 0, 10, 10, 0.9, 1, 3]]), np.zeros((0, 7))]
    lines, seq_info = track_scores.mot_files(images, tracked)
    assert seq_info == {"a": 1, "b": 1}
    assert lines[0, "gt", "a"] == ["1,7,0.0,0.0,10.0,10.0,1,1,1"]
    assert lines[1, "gt", "b"] == ["1,9,5.0,5.0,10.0,10.0,1,1,1"]
    assert lines[1, "trk", "a"] == ["1,3,0.0,0.0,10.0,10.0,0.9,-1,-1,-1"]


def _tracks_from_gt(images: list[visdrone_vid.Image]) -> list[np.ndarray]:
    return [
        np.column_stack(
            [
                im.boxes[:, :2],
                im.boxes[:, :2] + im.boxes[:, 2:],
                np.ones(len(im.boxes)),
                visdrone_vid.merged(im.categories),
                im.track_ids,
            ]
        )
        for im in images
    ]


def test_trackeval_scores_perfect_tracks_and_counts_id_switches() -> None:
    pytest.importorskip("trackeval")
    images = [
        _image(f"a{f}", [1, 4], [[f, 0, 10, 10], [50, 50 + f, 10, 10]], [1, 2], "a")
        for f in range(4)
    ]
    perfect = track_scores.readings(images, _tracks_from_gt(images))
    assert perfect["track/HOTA"] == pytest.approx(1.0) and perfect["track/IDSW"] == 0
    assert perfect["track/two-three-wheeler/HOTA"] == 0  # absent here, so det-averaging matters

    switched = _tracks_from_gt(images)
    for rows in switched[2:]:
        rows[:, 6] += 10
    out = track_scores.readings(images, switched)
    assert out["track/IDSW"] == 2 and out["track/DetA"] == pytest.approx(1.0)
    assert out["track/HOTA"] < 1
