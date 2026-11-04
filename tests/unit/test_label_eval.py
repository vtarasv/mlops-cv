"""The labeling-trial harness's pure parts: the ignored-region rule and the VisDrone-rule splits,
tiling, the stage runner, the key-frame tracking rules and the TrackEval glue."""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from pydantic import ValidationError

from mlops_cv.autolabel import labeler
from mlops_cv.autolabel.labeler import Labels
from mlops_cv.config.settings import BoxFilter
from mlops_cv.data.subset import MANIFEST_FIELDS, csv_line, image_relpath, label_relpath
from mlops_cv.tuning.autolabel_visdrone_vid import (
    label_eval,
    open_vocab,
    track_scores,
    tracks,
    visdrone_vid,
)
from mlops_cv.tuning.autolabel_visdrone_vid.label_eval import LabelingConfig
from mlops_cv.tuning.autolabel_visdrone_vid.tracks import Tracker

STAGES = Path(__file__).parents[2] / "configs" / "tuning" / "autolabel-visdrone-vid"


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
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    """A raw val sequence of 22 frames and a subset holding frames 1 and 21 of it."""
    raw, subset = tmp_path / "raw", tmp_path / "subset"
    root = raw / "VisDrone2019-VID-val"
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
        "2,1,61,10,20,10,1,4,0,0",  # frame 2 is not in the subset
        "21,1,62,10,20,10,1,4,0,0",
    ]
    (root / "annotations" / "uav1_v.txt").write_text("\n".join(rows) + "\n")
    lines = [",".join(MANIFEST_FIELDS)]
    for frame, n in ((1, 2), (21, 1)):
        name = f"uav1_v_{frame:07d}"
        row = dict.fromkeys(MANIFEST_FIELDS, 0) | {"split": "val", "sequence": "uav1_v"}
        lines.append(csv_line(row | {"frame_index": frame}, MANIFEST_FIELDS))
        for rel in (image_relpath("val", name), label_relpath("val", name)):
            (subset / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 50)).save(subset / image_relpath("val", name))
        (subset / label_relpath("val", name)).write_text("1 0 0 0 0\n" * n)
    (subset / "manifest.csv").write_text("\n".join(lines) + "\n")
    return raw, subset


def test_read_split_reads_the_subset_frames_through_the_rule(dirs: tuple[Path, Path]) -> None:
    raw, subset = dirs
    first, last = visdrone_vid.read_split(raw, subset, "val")
    assert (first.name, last.name) == ("uav1_v_0000001.jpg", "uav1_v_0000021.jpg")
    assert first.path == subset / "images/val/uav1_v_0000001.jpg"
    assert first.categories.tolist() == [4] and first.track_ids.tolist() == [1]
    assert last.boxes.tolist() == [[62, 10, 20, 10]]


def test_clips_read_every_raw_frame(dirs: tuple[Path, Path]) -> None:
    images = visdrone_vid.read_split(*dirs, "val", clips=True)
    assert [im.frame for im in images] == list(range(1, 23))
    assert images[0].path.parent.name == "uav1_v"


def test_rule_split_links_frames_and_drops_ignored_boxes(dirs: tuple[Path, Path]) -> None:
    raw, subset = dirs
    assert visdrone_vid.build_rule_split(raw, subset, "val") == (2, 1)
    label = (subset / "labels/val-visdrone-rule/uav1_v_0000001.txt").read_text()
    assert label == "1 0.700000 0.300000 0.200000 0.200000\n"
    link = subset / "images/val-visdrone-rule/uav1_v_0000001.jpg"
    assert link.readlink() == Path("../val/uav1_v_0000001.jpg") and link.exists()


def test_predictions_go_through_the_rule() -> None:
    image = dataclasses.replace(_image("a", []), ignored=np.array([[0, 0, 50, 100]]))
    rows = np.array([[0, 0, 10, 10, 0.9, 0], [60, 0, 70, 10, 0.8, 1]], np.float32)
    assert visdrone_vid.drop_all_ignored({"a": rows}, [image])["a"].tolist() == [rows[1].tolist()]


def test_slice_grid_matches_sahi() -> None:
    grid = open_vocab.slice_grid(1360, 765, 640)
    assert sorted({x for x, *_ in grid}) == [0, 512, 720]
    assert sorted({y for _, y, *_ in grid}) == [0, 125]
    assert all(x1 - x0 == 640 and y1 - y0 == 640 for x0, y0, x1, y1 in grid)
    assert open_vocab.slice_grid(500, 300, 640) == [(0, 0, 500, 300)]


def test_tiled_shifts_each_pass_and_adds_the_full_frame_unless_left_out() -> None:
    def detect(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        return np.array([[1, 2, 3, 4, 0.5, 1]], np.float32), ["car"]

    image = np.zeros((10, 16, 3), np.uint8)
    rows, labels, windows = open_vocab.tiled(detect, image, 10)
    assert windows.tolist() == [[0, 0, 10, 10], [6, 0, 16, 10], [0, 0, 16, 10]]
    assert rows[:, :2].tolist() == [[1, 2], [7, 2], [1, 2]] and labels == ["car"] * 3
    assert open_vocab.tiled(detect, image, 20)[2].tolist() == [[0, 0, 16, 10]]
    slices = open_vocab.tiled(detect, image, 10, full_frame=False)[2]
    assert slices.tolist() == [[0, 0, 10, 10], [6, 0, 16, 10]]


def test_edge_rule_drops_boxes_cut_by_interior_slice_edges_only() -> None:
    boxes = np.array([[50, 10, 59, 20], [0, 0, 20, 20], [50, 10, 59, 20]], np.float32)
    windows = np.array([[0, 0, 60, 60], [0, 0, 60, 60], [0, 0, 100, 100]], np.float32)
    assert open_vocab.cut_by_slice_edge(boxes, windows, 100, 100).tolist() == [True, False, False]


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
    keep, conf = open_vocab.merge_passes(rows, windows, "edge", 100, 100)
    assert keep.tolist() == [1, 3] and conf.tolist() == pytest.approx([0.6, 0.4])


def test_contained_merge_keeps_the_larger_box_with_the_pair_top_score() -> None:
    pytest.importorskip("torchvision")
    rows = np.array(
        [[50, 10, 59, 20, 0.9, 1], [40, 10, 70, 20, 0.6, 1], [50, 10, 59, 20, 0.8, 0]], np.float32
    )
    windows = np.array([[0, 0, 60, 60], [0, 0, 100, 100], [0, 0, 60, 60]], np.float32)
    keep, conf = open_vocab.merge_passes(rows, windows, "contained", 100, 100)
    assert keep.tolist() == [1, 2] and conf.tolist() == pytest.approx([0.9, 0.8])


def test_contained_merge_keeps_a_whole_box_inside_a_larger_one() -> None:
    pytest.importorskip("torchvision")
    # a car (uncut by its slice) inside a frame-sized vehicle box is no cut-off piece
    rows = np.array([[40, 10, 52, 20, 0.9, 1], [0, 0, 100, 100, 0.1, 1]], np.float32)
    windows = np.array([[0, 0, 60, 60], [0, 0, 100, 100]], np.float32)
    keep, conf = open_vocab.merge_passes(rows, windows, "contained", 100, 100)
    assert keep.tolist() == [1, 0] and conf.tolist() == pytest.approx([0.1, 0.9])


def test_a_tiled_trial_merges_before_the_box_filter() -> None:
    pytest.importorskip("torchvision")
    detector = open_vocab.Detector(tiling={"side": 60})
    config = LabelingConfig(detector=detector, box_filter={"min_side": None, "max_area": None})
    labeler = open_vocab.TrialLabeler(config.detector, config.box_filter, "cpu", 300, detect=False)
    rows = np.array([[40, 10, 59, 20, 0.9, 1], [40, 10, 70, 20, 0.6, 1]], np.float32)
    windows = np.array([[0, 0, 60, 60], [0, 0, 100, 100]], np.float32)
    out = labeler.finish(np.zeros((100, 100, 3)), Labels(rows, ["a", "b"], {}, 2, windows=windows))
    assert out.labels == ["b"] and set(out.ms) == {"merge", "filter"}


def test_trial_detectors_beyond_production_are_picked_by_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module, name in ((open_vocab, "owlv2"), (open_vocab, "yoloe"), (labeler, "grounding_dino")):
        monkeypatch.setattr(module, name, lambda cfg, device, max_det, name=name: name)
    pick = {
        "IDEA-Research/grounding-dino-base": "grounding_dino",
        "google/owlv2-base-patch16-ensemble": "owlv2",
        "yoloe-26l-seg.pt": "yoloe",
    }
    for model, loader in pick.items():
        assert open_vocab.detector(open_vocab.Detector(model=model), "cpu", 300) == loader


def _config(**sections: object) -> LabelingConfig:
    return LabelingConfig.model_validate({"detector": {"model": "m"}, **sections})


def test_the_store_is_keyed_by_frame_set_and_detector_but_not_the_merge() -> None:
    def store(c: LabelingConfig, frames: str = "train") -> Path:
        return label_eval.detections_store(frames, c)

    assert store(_config()).parent == label_eval.RUNS_DIR / "detections"
    assert store(_config()).name.startswith("train-")
    assert store(_config(), "train-clips") != store(_config())
    assert store(_config(detector={"model": "m", "imgsz": 1280})) != store(_config())
    halves = [_config(detector={"model": "m", "dtype": d}) for d in ("bfloat16", None)]
    assert len({store(c) for c in [_config(), *halves]}) == 3
    edge, contained, other = (
        _config(detector={"model": "m", "tiling": {"side": side, "merge": merge}})
        for side, merge in [(806, "edge"), (806, "contained"), (504, "edge")]
    )
    assert store(edge) == store(contained) != store(other)
    slices = _config(detector={"model": "m", "tiling": {"side": 806, "full_frame": False}})
    assert store(slices) != store(edge)


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
    trials = label_eval.load_stage(_stage(tmp_path, stage))
    a, b = trials["s-a"], trials["s-b"]
    assert a.detector.threshold == 0.1 and len(a.detector.prompts) == 2
    assert b.detector.prompts == {"vehicle": "vehicle"} and b.box_filter.min_score == 0.4
    assert b.box_filter == BoxFilter(min_score=0.4)  # unset knobs are production's


def test_stage_rejects_repeats_and_unknown_keys(tmp_path: Path) -> None:
    trial = {"name": "a", "detector": {"model": "m"}}
    with pytest.raises(ValueError, match="repeats"):
        label_eval.load_stage(_stage(tmp_path, {"trials": [trial, trial]}))
    with pytest.raises(ValidationError):
        label_eval.load_stage(
            _stage(tmp_path, {"trials": [{**trial, "box_filter": {"min_sid": 8}}]})
        )
    with pytest.raises(ValidationError):
        label_eval.load_stage(_stage(tmp_path, {"trials": [{**trial, "segmenter": "s"}]}))


@pytest.mark.parametrize("stage", sorted(STAGES.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_labeling_stage_loads(stage: Path) -> None:
    assert label_eval.load_stage(stage)


def _image(
    name: str,
    categories: list[int],
    boxes: list | None = None,
    track_ids: list[int] | None = None,
    sequence: str = "s",
    frame: int = 1,
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
        frame,
    )


def test_cap_subset_and_recall_ceiling() -> None:
    images = [_image("a", [4] * 250 + [1] * 10), _image("b", [4] * 50)]
    assert label_eval.uncapped(images, 200) == ["b"]
    assert label_eval.recall_ceiling(images, 200) == pytest.approx(1 - 50 / 310)


def test_select_frames_filters_sequences_then_samples_then_limits() -> None:
    images = [_image(f"{q}{i}", [], sequence=q) for q in "ab" for i in range(4)]

    def names(sequences: list[str] | None, every: int | None, limit: int | None) -> list[str]:
        return [im.name for im in label_eval.select_frames(images, sequences, every, limit)]

    assert names(None, None, None) == [im.name for im in images]
    assert names(None, 3, None) == ["a0", "a3", "b2"]
    assert names(["b"], 2, 1) == ["b0"]


def _labels(ms: float, rows: np.ndarray | None = None) -> Labels:
    rows = np.zeros((0, 6)) if rows is None else rows
    return Labels(rows, ["car"] * len(rows), {"detector": ms, "filter": 1.0}, len(rows))


def test_latency_leaves_out_warm_up() -> None:
    labels = [_labels(1000.0)] * label_eval.WARMUP + [_labels(10.0), _labels(20.0)]
    out = label_eval.latency(labels)
    assert out["latency/detector/p50_ms"] == 15.0
    assert out["latency/total/mean_ms"] == 16.0


def test_annotations_are_one_file_per_sequence(tmp_path: Path) -> None:
    rows = np.array([[1, 2, 5, 8, 0.5, 0]], np.float32)
    images = [_image("a", [], sequence="s1", frame=3), _image("b", [], sequence="s2")]
    label_eval.write_annotations(tmp_path, images, [_labels(1.0, rows), _labels(1.0)])
    assert (tmp_path / "s1.txt").read_text() == "3,-1,1.00,2.00,4.00,6.00,0.5,4,-1,-1\n"
    assert (tmp_path / "s2.txt").read_text() == ""


def test_stored_detections_round_trip_and_check_their_frames(tmp_path: Path) -> None:
    images = [_image("a", []), _image("b", [])]
    rows = np.array([[1, 2, 5, 8, 0.5, 0]], np.float32)
    raw = [
        Labels(rows, ["car"], {"detector": 7.0}, 1),
        Labels(np.zeros((0, 6)), [], {"detector": 3.0}, 0),
    ]
    path = tmp_path / "d.npz"
    label_eval.save_detections(path, images, raw)

    a, b = label_eval.load_detections(path, images)
    assert a.rows.tolist() == rows.tolist() and a.labels == ["car"] and a.ms == {"detector": 7.0}
    assert (len(b.rows), b.detected, b.ms) == (0, 0, {"detector": 3.0})
    assert a.windows is None
    (only_b,) = label_eval.load_detections(path, images[1:])  # a subset reads by frame name
    assert only_b.ms == {"detector": 3.0}
    with pytest.raises(ValueError, match="lacks c"):
        label_eval.load_detections(path, [*images, _image("c", [])])


def test_stored_tiled_detections_keep_their_windows(tmp_path: Path) -> None:
    rows = np.array([[1, 2, 5, 8, 0.5, 0]], np.float32)
    windows = np.array([[0, 0, 10, 10]], np.float32)
    empty = np.zeros((0, 6), np.float32)
    raw = [
        Labels(rows, ["car"], {"detector": 7.0}, 1, windows=windows),
        Labels(empty, [], {"detector": 1.0}, 0, windows=np.zeros((0, 4), np.float32)),
    ]
    images = [_image("a", []), _image("b", [])]
    path = tmp_path / "d.npz"
    label_eval.save_detections(path, images, raw)
    a, b = label_eval.load_detections(path, images)
    assert a.windows is not None and a.windows.tolist() == windows.tolist()
    assert b.windows is not None and b.windows.shape == (0, 4)


def test_amortised_latency_drops_whole_key_frame_cycles() -> None:
    key = Labels(np.zeros((0, 6)), [], {"detector": 100.0, "tracker": 10.0}, 0)
    tracked = Labels(np.zeros((0, 6)), [], {"tracker": 10.0}, 0)
    labels = ([key] + [tracked] * 4) * 3  # K = 5; the first cycle is the warm-up
    out = label_eval.amortised_latency(labels, Tracker(model="m", key_interval=5))
    assert out["latency/detector/amortised_ms"] == 20.0
    assert out["latency/total/amortised_ms"] == 30.0


def test_match_assigns_one_to_one_and_every_unmatched_box_is_new() -> None:
    pytest.importorskip("lap")
    iou = np.array([[0.9, 0.0], [0.6, 0.1], [0.0, 0.4], [0.1, 0.2]])
    pairs, new = tracks.match(iou, 0.3)
    assert pairs == {0: 0, 2: 1}
    assert new.tolist() == [1, 3]  # box 1 lost track 0 to box 0, so it starts a track


def test_match_maximises_the_total_iou_not_the_best_pair() -> None:
    pytest.importorskip("lap")
    pairs, _ = tracks.match(np.array([[0.9, 0.8], [0.85, 0.0]]), 0.3)
    assert pairs == {0: 1, 1: 0}  # greedy would take 0.9 and leave box 1 without its track


def test_match_without_tracks_makes_every_box_new() -> None:
    pytest.importorskip("lap")
    pairs, new = tracks.match(np.zeros((2, 0)), 0.3)
    assert pairs == {} and new.tolist() == [0, 1]


def test_the_younger_duplicate_ends() -> None:
    iou = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.9], [0.0, 0.9, 1.0]])
    assert tracks.younger_duplicates(iou, [4, 2, 7], 0.8) == {4, 7}


def _rows(*scores: float) -> np.ndarray:
    return np.array([[0, 0, 1, 1, s, 1] for s in scores], float).reshape(-1, 6)


def test_key_frames_start_match_and_end_tracks() -> None:
    pytest.importorskip("lap")
    live = tracks.Tracks(Tracker(model="m", unmatched_keys=2))
    ended, boxes = live.key_frame([], _rows(0.9, 0.5), ["a", "b"], np.zeros((2, 0)), np.eye(0))
    assert ended == set() and boxes == {1: 0, 2: 1}
    assert live.live[2].label == "b" and live.live[2].cls == 1

    # Track 1 matches a fresher box, track 2 misses its first key frame.
    iou = np.array([[0.8, 0.0]])
    ended, boxes = live.key_frame([1, 2], _rows(0.6), ["a"], iou, np.eye(2))
    assert ended == set() and boxes == {1: 0} and live.live[1].score == 0.6
    ended, _ = live.key_frame([1, 2], _rows(0.6), ["a"], iou, np.eye(2))
    assert ended == {2} and list(live.live) == [1]


def test_duplicates_end_before_matching_and_the_cap_keeps_the_best_new_boxes() -> None:
    pytest.importorskip("lap")
    live = tracks.Tracks(Tracker(model="m", max_objects=3))
    live.key_frame([], _rows(0.9, 0.8), ["a", "a"], np.zeros((2, 0)), np.eye(0))
    track_iou = np.array([[1.0, 0.95], [0.95, 1.0]])
    box_iou = np.array([[0.1, 0.9], [0.0, 0.0], [0.0, 0.0]])  # box 0 fits track 2 only
    ended, boxes = live.key_frame([1, 2], _rows(0.9, 0.7, 0.6), ["a"] * 3, box_iou, track_iou)
    assert ended == {2}
    assert boxes == {3: 0, 4: 1}  # box 0 is new, as its only match ended; the cap drops box 2
    assert live.live[1].missed == 1


def test_empty_masks_end_a_track() -> None:
    pytest.importorskip("lap")
    live = tracks.Tracks(Tracker(model="m", empty_frames=2))
    live.key_frame([], _rows(0.9), ["a"], np.zeros((1, 0)), np.eye(0))
    assert live.observe([1], [0]) == set()
    assert live.observe([1], [5]) == set()  # a non-empty frame resets the count
    assert live.observe([1], [0]) == set()
    assert live.observe([1], [0]) == {1} and not live.live


def test_tight_boxes_bound_each_mask_and_zero_an_empty_one() -> None:
    torch = pytest.importorskip("torch")
    masks = torch.zeros((2, 6, 8), dtype=torch.bool)
    masks[0, 1:3, 2:7] = True
    assert tracks.tight_boxes(masks).tolist() == [[2, 1, 7, 3], [0, 0, 0, 0]]
    assert tracks.tight_boxes(masks[:0]).shape == (0, 4)


def test_box_iou_keeps_its_shape_when_empty() -> None:
    pytest.importorskip("torchvision")
    a = np.array([[0, 0, 2, 2]], float)
    assert tracks.box_iou(a, np.array([[1, 0, 3, 2]], float)).tolist() == [[pytest.approx(1 / 3)]]
    assert tracks.box_iou(a, np.zeros((0, 4))).shape == (1, 0)


@pytest.mark.parametrize("tracker", tracks.BOX_TRACKERS)
def test_box_trackers_take_the_floor_as_their_track_thresholds(
    tmp_path: Path, tracker: str
) -> None:
    pytest.importorskip("lap")
    images = []
    for i in range(5):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (100, 100)).save(path)
        images.append(dataclasses.replace(_image(path.name, []), path=path))
    rows = np.array([[10, 10, 30, 30, 0.5, 1]], np.float32)
    detections = [Labels(rows, ["car"], {"detector": 1.0}, 1)] * 5

    def tracked(floor: float | None) -> int:
        out = tracks.box_tracks(tracker, images, detections, floor)
        return sum(len(lab.rows) for lab, _ in out)

    assert tracked(0.4) > 0
    assert tracked(0.6) == 0  # 0.5 is under the floor: it starts no track


def test_box_trackers_share_one_pass_across_trackers_and_floors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passes = []
    monkeypatch.setattr(label_eval, "label_frames", lambda *a: passes.append(1) or [])
    monkeypatch.setattr(tracks, "box_tracks", lambda *a: [])
    detections: dict[str, list[Labels]] = {}
    for tracker in tracks.BOX_TRACKERS:
        for floor in (0.5, 0.6):
            config = _config(tracker={"model": tracker}, confidence_floor=floor)
            label_eval.label_sequences(config, [], "cpu", detections, None)
    gt = _config(tracker={"model": "botsort"}, ground_truth=True)
    label_eval.label_sequences(gt, [], "cpu", detections, None)
    assert len(passes) == 1  # the ground-truth trial runs no detector


def test_ground_truth_boxes_are_detections_at_score_one() -> None:
    im = _image("a", [1, 4], boxes=[[10, 20, 5, 8], [0, 0, 30, 10]])
    (lab,) = label_eval.ground_truth_labels([im])
    assert lab.rows.tolist() == [[10, 20, 15, 28, 1, 0], [0, 0, 30, 10, 1, 1]]
    assert lab.labels == ["pedestrian", "car"] and lab.detected == 2


@pytest.mark.parametrize("tracker", [None, {"model": "yonigozlan/EdgeTAM-hf"}])
def test_ground_truth_boxes_need_a_box_tracker(tracker: dict | None) -> None:
    with pytest.raises(ValidationError, match="box tracker"):
        _config(tracker=tracker, ground_truth=True)


def test_subset_positions_pick_the_subset_frames_inside_the_clips() -> None:
    images = [_image(f"{q}{f}", [], sequence=q, frame=f) for q in "ab" for f in (1, 2, 3)]
    frames = {"a": [2], "b": [1, 3], "c": [1]}
    assert label_eval.subset_positions(images, frames) == [1, 3, 5]


class _Session:
    output_dict_per_obj: dict[int, typing.Any]

    def __init__(self, ids: list[int]) -> None:
        self._obj_id_to_idx = {o: i for i, o in enumerate(ids)}
        self._obj_idx_to_id = dict(enumerate(ids))
        self.obj_ids = list(ids)
        self.obj_with_new_inputs: list[int] = []
        self.processed_frames = {30: "frame"}
        for store in tracks.PER_OBJECT:
            setattr(self, store, {i: f"{store}-{o}" for i, o in enumerate(ids)})


def test_set_objects_reindexes_the_session() -> None:
    session = _Session([5, 6, 7])
    state = tracks.objects(session)
    tracks.set_objects(session, {o: s for o, s in state.items() if o != 6})
    assert session.obj_ids == [5, 7] and session._obj_idx_to_id == {0: 5, 1: 7}
    assert session.output_dict_per_obj == {
        0: "output_dict_per_obj-5",
        1: "output_dict_per_obj-7",
    }
    tracks.set_objects(session, {})
    tracks.set_objects(session, state)
    assert tracks.objects(session) == state


def test_prune_keeps_seed_frames_and_the_last_window() -> None:
    session = _Session([1])
    frames = {t: {"high_res_masks": 1} for t in range(31)}
    out = {"cond_frame_outputs": {0: {"high_res_masks": 1}}, "non_cond_frame_outputs": frames}
    session.output_dict_per_obj = {0: out}
    tracks.prune(session, 30)
    assert sorted(out["non_cond_frame_outputs"]) == list(range(30 - tracks.KEEP_FRAMES, 31))
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
    assert {"track/DetA", "track/AssA", "track/IDF1"} <= perfect.keys()

    switched = _tracks_from_gt(images)
    for rows in switched[2:]:
        rows[:, 6] += 10
    out = track_scores.readings(images, switched)
    assert out["track/IDSW"] == 2 and out["track/DetA"] == pytest.approx(1.0)
    assert out["track/HOTA"] < 1
