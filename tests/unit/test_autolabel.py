"""The production auto-labeler's pure parts: settings, box filter, top rows, annotation lines."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlops_cv.autolabel import labeler
from mlops_cv.autolabel.labeler import Labels
from mlops_cv.config.settings import AutolabelDetector, AutolabelSettings, BoxFilter
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
