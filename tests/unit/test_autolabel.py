"""The production auto-labeler's parts: settings, box filter, top rows, tracker, annotations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from mlops_cv.autolabel import labeler
from mlops_cv.autolabel.labeler import Labels
from mlops_cv.config.settings import (
    AutolabelDetector,
    AutolabelSettings,
    AutolabelTracker,
    BoxFilter,
)
from mlops_cv.data.convert_visdrone_vid import (
    convert_sequence_text,
)
from mlops_cv.tuning.common import CONF, MAX_DET

STAGES = Path(__file__).parents[2] / "configs" / "tuning" / "autolabel-visdrone-vid"


def test_prompts_name_merged_classes_only() -> None:
    assert AutolabelDetector(prompts={"car": "vehicle"}).prompts == {"car": "vehicle"}
    with pytest.raises(ValueError, match="not a merged class"):
        AutolabelDetector(prompts={"car": "automobile"})


def test_the_detector_scores_like_val() -> None:
    assert AutolabelSettings().detector.threshold == CONF
    assert AutolabelSettings().max_det == MAX_DET


def test_defaults_are_the_labeler_stage() -> None:
    from mlops_cv.tuning.autolabel_visdrone_vid.label_eval import load_stage

    (trial,) = load_stage(STAGES / "labeler.yaml").values()
    settings = AutolabelSettings()
    assert trial.detector.model_dump(include=set(type(settings.detector).model_fields)) == (
        settings.detector.model_dump()
    )
    assert trial.box_filter == settings.box_filter
    assert trial.confidence_floor == settings.confidence_floor
    assert trial.tracker is not None
    assert trial.tracker.model_dump(include=set(AutolabelTracker.model_fields)) == (
        settings.tracker.model_dump()
    )


def test_the_tracker_is_tracktrack_and_the_floor_sets_its_thresholds() -> None:
    assert AutolabelSettings().tracker == AutolabelTracker(model="tracktrack", overrides={})
    with pytest.raises(ValueError, match="confidence floor sets"):
        AutolabelTracker(overrides={"track_high_thresh": 0.5})


def test_tracker_args_are_ultralytics_yaml_under_overrides_and_the_floor() -> None:
    pytest.importorskip("ultralytics")
    args = labeler.tracker_args(AutolabelTracker(overrides={"match_thresh": 0.8}), 0.3)
    assert (args["match_thresh"], args["track_high_thresh"], args["new_track_thresh"]) == (
        0.8,
        0.3,
        0.3,
    )
    assert labeler.tracker_args(AutolabelTracker(), None)["new_track_thresh"] == 0.7
    with pytest.raises(ValueError, match="track_buffr"):
        labeler.tracker_args(AutolabelTracker(overrides={"track_buffr": 60}), 0.3)


def test_a_default_tracked_trial_reproduces_the_production_labeler(tmp_path: Path) -> None:
    pytest.importorskip("lap")
    from mlops_cv.tuning.autolabel_visdrone_vid import label_eval, visdrone_vid

    rng = np.random.default_rng(0)
    start = rng.uniform(0, 60, (4, 2))
    images, raw = [], []
    for frame in range(1, 9):
        path = tmp_path / f"{frame:07d}.jpg"
        Image.new("RGB", (100, 100)).save(path)
        none = np.zeros((0, 4))
        images.append(
            visdrone_vid.Image(
                path.name, path, 100, 100, none, none[:, 0], none[:, 0], none, "s", frame
            )
        )
        xy = start + 2 * frame  # four boxes drifting, scored either side of the floor
        rows = np.column_stack([xy, xy + 20, rng.uniform(0.1, 0.9, 4), np.ones(4)])
        raw.append(Labels(rows.astype(np.float32), ["car"] * 4, {"detector": 1.0}, 4))

    settings = AutolabelSettings()
    production = labeler.Labeler(settings.detector, settings.box_filter, "cpu", 300, detect=False)
    passes = iter([(r.rows, r.labels) for r in raw])
    production.detect = lambda image: next(passes)
    frames = (labeler.read_rgb(im.path) for im in images)
    lines = list(
        labeler.label_clip(production, settings.tracker, settings.confidence_floor, frames)
    )

    (trial,) = label_eval.load_stage(STAGES / "labeler.yaml").values()
    store = tmp_path / "pass.npz"
    label_eval.save_detections(store, images, raw)
    tracked, _ = label_eval.label_sequences(trial, images, "cpu", {}, store)
    label_eval.write_annotations(tmp_path, images, tracked)

    assert {line.split(",")[1] for line in lines} - {"-1"}  # tracks were written
    assert min(float(line.split(",")[6]) for line in lines) >= settings.confidence_floor
    assert lines == (tmp_path / "s.txt").read_text().splitlines()


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
    f = BoxFilter(min_side=8, max_area=0.5, min_score=0.4, nms_iou=None, max_objects=1)
    assert labeler.filter_boxes(rows, f, 100, 100).tolist() == [4]
    off = BoxFilter(min_side=None, max_area=None, nms_iou=None)
    assert labeler.filter_boxes(rows, off, 100, 100).tolist() == [0, 1, 4, 3, 2]


def test_filter_nms_is_class_agnostic() -> None:
    pytest.importorskip("torchvision")
    rows = np.array([[0, 0, 10, 10, 0.9, 0], [0, 0, 10, 11, 0.8, 1]], np.float32)
    assert labeler.filter_boxes(rows, BoxFilter(nms_iou=0.6), 100, 100).tolist() == [0]


def test_top_rows_keep_the_best_boxes_above_the_floor() -> None:
    torch = pytest.importorskip("torch")
    cfg = labeler.AutolabelDetector(prompts={"car": "vehicle", "person": "person"}, threshold=0.1)
    boxes = torch.arange(4.0)[:, None] + torch.tensor([0.0, 0, 1, 1])
    conf = torch.tensor([0.05, 0.9, 0.5, 0.7])
    rows, labels = labeler.top_rows(boxes, conf, torch.tensor([0, 1, 0, 0]), cfg, 2)
    assert rows[:, 0].tolist() == [1, 3] and labels == ["person", "car"]
    assert rows[:, 5].tolist() == [labeler.MERGED["person"], labeler.MERGED["vehicle"]]
    empty, _ = labeler.top_rows(boxes, conf * 0, torch.tensor([0, 1, 0, 0]), cfg, 2)
    assert empty.shape == (0, 6)


def _rows(*scores: float) -> np.ndarray:
    return np.array([[0, 0, 1, 1, s, 1] for s in scores], float).reshape(-1, 6)


def test_the_floor_keeps_rows_at_or_above_it() -> None:
    labels = Labels(
        _rows(0.9, 0.5, 0.6), ["a", "b", "c"], {"detector": 1.0}, 3, np.array([4, 5, 6])
    )
    kept = labeler.above(labels, 0.6)
    assert kept.rows[:, 4].tolist() == [0.9, 0.6] and kept.labels == ["a", "c"]
    assert kept.track_ids is not None and kept.track_ids.tolist() == [4, 6]
    assert (kept.ms, kept.detected) == ({"detector": 1.0}, 3)
    assert labeler.above(labels, None) is labels


def test_raw_category_is_the_prompt_or_its_merged_class() -> None:
    assert labeler.raw_category("van", 1) == 5
    assert [labeler.raw_category("person", 0), labeler.raw_category("vehicle", 1)] == [1, 4]
    assert labeler.raw_category("motorcycle", 2) == 3


def test_annotation_lines_golden() -> None:
    rows = np.array([[10, 20, 40, 30, 0.5972, 1], [1, 2, 3, 4, 0.25, 0]], np.float32)
    labels = Labels(rows, ["car", "person"], {}, 2)
    assert labeler.annotation_lines(7, labels) == [
        "7,-1,10.00,20.00,30.00,10.00,0.5972,4,-1,-1",
        "7,-1,1.00,2.00,2.00,2.00,0.25,1,-1,-1",
    ]
    tracked = Labels(rows[:1], ["car"], {}, 1, np.array([3]))
    assert labeler.annotation_lines(7, tracked)[0].startswith("7,3,")


def test_ingestion_reads_annotation_lines_back() -> None:
    rows = np.array([[10, 20, 40, 30, 0.6, 1]], np.float32)
    text = "\n".join(labeler.annotation_lines(1, Labels(rows, ["truck"], {}, 1)))
    (box,) = convert_sequence_text(text, (100, 100))[1]
    assert (box.cls, box.xc, box.yc, box.w, box.h) == pytest.approx((1, 0.25, 0.25, 0.3, 0.1))


def test_grounding_dino_unties_per_layer_decoder_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = SimpleNamespace(cls_token_id=1, sep_token_id=2, convert_tokens_to_ids=lambda t: 3)
    tied = {}

    class Model:
        def to(self, device: str) -> Model:
            return self

        def eval(self) -> Model:
            return self

    def model(name: str, tie_word_embeddings: bool) -> Model:
        tied[name] = tie_word_embeddings
        return Model()

    processor = SimpleNamespace(tokenizer=tokenizer)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda _: processor)
    monkeypatch.setattr(transformers.AutoModelForZeroShotObjectDetection, "from_pretrained", model)
    for name, share in [("base", True), ("mm-large", False)]:
        config = SimpleNamespace(decoder_bbox_embed_share=share)
        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda _, c=config: c)
        labeler.grounding_dino(AutolabelDetector(model=name), "cpu", 300)
    assert tied == {"base": True, "mm-large": False}


def test_prompt_spans_split_at_separators() -> None:
    spans = labeler.prompt_spans([101, 5, 1012, 7, 8, 1012, 102], {101, 102, 1012})
    assert spans == [[1], [3, 4]]


def test_the_detector_is_picked_by_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("grounding_dino", "sam3"):
        monkeypatch.setattr(labeler, name, lambda cfg, device, max_det, name=name: name)
    pick = {"IDEA-Research/grounding-dino-base": "grounding_dino", "facebook/sam3": "sam3"}
    for model, loader in pick.items():
        assert labeler.detector(AutolabelDetector(model=model), "cpu", 300) == loader


def test_autocast_runs_matmuls_at_the_dtype_and_is_off_without_one() -> None:
    torch = pytest.importorskip("torch")
    x = torch.ones(2, 2)
    with labeler.autocast("bfloat16", "cpu"):
        assert (x @ x).dtype == torch.bfloat16
    with labeler.autocast(None, "cpu"):
        assert (x @ x).dtype == torch.float32


def test_the_command_writes_one_annotation_file_per_sequence(monkeypatch, tmp_path: Path) -> None:
    from mlops_cv.autolabel import __main__ as cli

    seq = tmp_path / "sequences" / "uav0000001_00000_v"
    seq.mkdir(parents=True)
    for i in (1, 2):
        Image.new("RGB", (8, 8)).save(seq / f"{i:07d}.jpg")
    monkeypatch.setattr(labeler, "Labeler", lambda *a, **k: None)
    monkeypatch.setattr(
        labeler,
        "label_clip",
        lambda _l, _t, _f, frames: (f"{i},1" for i, _ in enumerate(frames, 1)),
    )
    out = tmp_path / "out"

    assert cli.main([str(seq), "--out", str(out)]) == 0
    assert (out / "uav0000001_00000_v.txt").read_text() == "1,1\n2,1\n"
    assert not list(out.glob("*.part"))

    (seq / "0000002.jpg").rename(seq / "0000003.jpg")
    assert cli.main([str(seq), "--out", str(out)]) == 2  # a frame gap
