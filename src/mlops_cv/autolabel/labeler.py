"""The labeling config and the auto-labeler it builds: one decoded image -> boxes and RLE masks.

Every model loads through transformers, except the champion's YOLO26 (ultralytics). Each stage
is timed on its own, the GPU synchronised at its edges.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from mlops_cv.data.convert_visdrone_vid import YOLO_NAMES

MergedClass = Literal["person", "vehicle", "two-three-wheeler"]
MERGED = {name: c for c, name in YOLO_NAMES.items()}

# The champion's YOLO26, resolved from the registry: the supervised reference row.
CHAMPION = "champion"
# Boxes per SAM 2 mask-decoder pass: bounds the full-resolution mask upsample's memory.
SEGMENT_CHUNK = 32
# ultralytics' box trackers, run with their default settings; any other tracker is a video model.
BOX_TRACKERS = ("botsort", "bytetrack")

Detect = Callable[[np.ndarray], tuple[np.ndarray, list[str]]]
Segment = Callable[[np.ndarray, np.ndarray], list[dict[str, Any]]]


class Detector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str  # a transformers repo id, or ``champion``
    prompts: dict[str, MergedClass] = {}  # prompt text -> the merged class its boxes get
    threshold: float = 0.25  # the detector's own score floor


class BoxFilter(BaseModel):
    """The box filter's stages, applied in this order; ``None`` turns a stage off."""

    model_config = ConfigDict(extra="forbid")

    min_side: float | None = None  # px
    max_area: float | None = None  # share of the frame
    min_score: float | None = None
    nms_iou: float | None = None  # class-agnostic
    nested_ios: float | None = None  # drop a box this share inside a larger kept box
    max_objects: int | None = None  # the highest-scoring K


class Tracker(BaseModel):
    """Sequence mode. The key-frame rules apply to the video trackers, not to ``BOX_TRACKERS``."""

    model_config = ConfigDict(extra="forbid")

    model: str  # one of ``BOX_TRACKERS``, or a transformers video repo (EdgeTAM, SAM 2.1)
    key_interval: int = 10  # K: frame 0 and every K-th frame after it are key frames
    new_iou: float = 0.3  # tau: a kept box below it against every live track is a new object
    empty_frames: int = 25  # a track ends after this many consecutive empty-mask frames
    unmatched_keys: int = 2  # ... or after this many consecutive key frames without a match
    duplicate_iou: float = 0.8  # of two tracks whose masks overlap above it, the younger ends
    max_objects: int | None = None  # live-track cap; new objects beyond it aren't added


class LabelingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector: Detector
    box_filter: BoxFilter = BoxFilter()
    segmenter: str | None = None  # a transformers SAM 2 repo id; ``None`` keeps boxes only
    tracker: Tracker | None = None  # sequence mode; ``None`` labels each frame on its own

    @model_validator(mode="after")
    def _one_mask_source(self) -> LabelingConfig:
        if self.tracker and self.tracker.model not in BOX_TRACKERS and self.segmenter:
            raise ValueError(f"{self.tracker.model} makes its own masks: drop the segmenter")
        return self


@dataclass
class Labels:
    """One image's auto-labels: the kept detector rows and, with a segmenter, a mask per row."""

    rows: np.ndarray  # (n, 6) x1 y1 x2 y2 conf merged-cls, source pixels
    labels: list[str]  # the detector's label per row (a prompt, or a champion class)
    masks: list[dict[str, Any]] | None  # COCO compressed RLE per row
    ms: dict[str, float]  # wall-clock per stage
    detected: int  # detector boxes before the filter
    track_ids: np.ndarray | None = None  # sequence mode: the track per row


def ios(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Intersection over the smaller box's area, ``box`` (4,) against ``others`` (m, 4) xyxy."""
    lt = np.maximum(box[:2], others[:, :2])
    rb = np.minimum(box[2:], others[:, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    own = (box[2] - box[0]) * (box[3] - box[1])
    return inter / (np.minimum(own, area) + 1e-9)


def drop_nested(boxes: np.ndarray, threshold: float) -> np.ndarray:
    """The nested-box rule, largest box first: a box inside a kept larger one is dropped.

    It keeps the outer box, so a detector box drawn around a group loses the objects inside it.
    Returns the kept positions.
    """
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    for i in np.argsort(-area, kind="stable"):
        if not keep or ios(boxes[i], boxes[keep]).max() < threshold:
            keep.append(int(i))
    return np.array(keep, int)


def filter_boxes(rows: np.ndarray, f: BoxFilter, width: int, height: int) -> np.ndarray:
    """The indices of the ``rows`` that ``f`` keeps, highest score first."""
    x1, y1, x2, y2, conf = rows[:, :5].T
    w, h = x2 - x1, y2 - y1
    ok = np.ones(len(rows), bool)
    if f.min_side is not None:
        ok &= np.minimum(w, h) >= f.min_side
    if f.max_area is not None:
        ok &= w * h <= f.max_area * width * height
    if f.min_score is not None:
        ok &= conf >= f.min_score
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(-conf[idx], kind="stable")]
    if f.nms_iou is not None and len(idx):
        import torch
        from torchvision.ops import nms

        kept = nms(torch.from_numpy(rows[idx, :4]), torch.from_numpy(conf[idx]), f.nms_iou)
        idx = idx[kept.numpy()]
    if f.nested_ios is not None and len(idx):
        idx = idx[drop_nested(rows[idx, :4], f.nested_ios)]
        idx = idx[np.argsort(-conf[idx], kind="stable")]
    return idx[: f.max_objects]


def prompt_spans(input_ids: Sequence[int], separators: set[int]) -> list[list[int]]:
    """The token positions of each prompt in a ``a . b .`` text, split at the separator ids."""
    spans: list[list[int]] = [[]]
    for i, token in enumerate(input_ids):
        if token not in separators:
            spans[-1].append(i)
        elif spans[-1]:
            spans.append([])
    return [s for s in spans if s]


def _device(device: str) -> str:
    return f"cuda:{device}" if device.isdigit() else device


def grounding_dino(cfg: Detector, device: str) -> Detect:
    """Grounding DINO: each box goes to the prompt whose tokens score it highest.

    transformers' own post-processing decodes the tokens above a threshold into a phrase, which
    can run two prompts together; the per-prompt maximum can't.
    """
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor: Any = AutoProcessor.from_pretrained(cfg.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.model).to(device).eval()
    names = [p.strip().lower() for p in cfg.prompts]
    text = ". ".join(names) + "."
    classes = np.array([MERGED[c] for c in cfg.prompts.values()])
    tok = processor.tokenizer
    separators = {tok.cls_token_id, tok.sep_token_id, tok.convert_tokens_to_ids(".")}

    def detect(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        h, w = image.shape[:2]
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inputs)
        spans = prompt_spans(inputs["input_ids"][0].tolist(), separators)
        if len(spans) != len(names):
            raise ValueError(f"{len(spans)} token spans for {len(names)} prompts in {text!r}")
        probs = out.logits[0].sigmoid()
        conf, label = torch.stack([probs[:, s].max(-1).values for s in spans], 1).max(1)
        keep = conf > cfg.threshold
        cx, cy, bw, bh = (out.pred_boxes[0][keep] * torch.tensor([w, h, w, h], device=device)).T
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        label_np = label[keep].cpu().numpy()
        rows = np.column_stack([boxes.cpu().numpy(), conf[keep].cpu().numpy(), classes[label_np]])
        return rows.astype(np.float32), [names[i] for i in label_np]

    return detect


def champion(device: str) -> Detect:
    """The champion YOLO26 at its training imgsz, with val's floor, cap and one-to-many head."""
    from ultralytics import YOLO

    from mlops_cv.config import get_settings
    from mlops_cv.evaluation.score_detections import CONF, MAX_DET
    from mlops_cv.optimize.variants import SERVER
    from mlops_cv.tracking import champion as champion_mod
    from mlops_cv.tracking import client
    from mlops_cv.tracking.resolve import resolve_model, source_run_id

    uri = client.champion_uri(get_settings(), SERVER)
    weights, _ = resolve_model(uri)
    imgsz = champion_mod.training_imgsz(source_run_id(uri))
    model = YOLO(str(weights))

    def detect(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        bgr = np.ascontiguousarray(image[..., ::-1])
        result: Any = model.predict(
            bgr,
            imgsz=imgsz,
            device=device,
            conf=CONF,
            max_det=MAX_DET,
            end2end=False,
            verbose=False,
        )[0]
        rows = result.boxes.data.cpu().numpy()
        return rows, [YOLO_NAMES[int(c)] for c in rows[:, 5]]

    return detect


def rle(masks: Any) -> list[dict[str, Any]]:
    """``(n, H, W)`` bool torch masks -> COCO compressed RLE."""
    import torch
    from pycocotools import mask

    mask_utils: Any = mask
    hwn = masks.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    return mask_utils.encode(np.asfortranarray(hwn))


def sam2(repo: str, device: str) -> Segment:
    """SAM 2 image mode: one image embedding, then the mask decoder over the boxes in chunks."""
    import torch
    from transformers import Sam2Model, Sam2Processor

    processor: Any = Sam2Processor.from_pretrained(repo)
    model: Any = Sam2Model.from_pretrained(repo)
    model = model.to(device).eval()

    def segment(image: np.ndarray, boxes: np.ndarray) -> list[dict[str, Any]]:
        if not len(boxes):
            return []
        inputs = processor(images=image, return_tensors="pt").to(device)
        sizes = inputs["original_sizes"]
        rles: list[dict[str, Any]] = []
        with torch.inference_mode():
            embeddings = model.get_image_embeddings(inputs["pixel_values"])
            for start in range(0, len(boxes), SEGMENT_CHUNK):
                chunk = boxes[start : start + SEGMENT_CHUNK].tolist()
                prompt = processor(original_sizes=sizes, input_boxes=[chunk], return_tensors="pt")
                out = model(
                    image_embeddings=embeddings,
                    input_boxes=prompt["input_boxes"].to(device),
                    multimask_output=False,
                )
                rles += rle(processor.post_process_masks(out.pred_masks, sizes)[0][:, 0])
        return rles

    return segment


class Labeler:
    """A labeling config's models, loaded once; called on one decoded RGB image."""

    def __init__(self, config: LabelingConfig, device: str) -> None:
        self.config = config
        self.device = device = _device(device)
        self._sync: Callable[[], None] = lambda: None
        if device.startswith("cuda"):
            import torch

            self._sync = torch.cuda.synchronize
        self.detect = (
            champion(device)
            if config.detector.model == CHAMPION
            else grounding_dino(config.detector, device)
        )
        self.segment = sam2(config.segmenter, device) if config.segmenter else None

    def timed(self, fn: Callable[..., Any], *args: Any) -> tuple[Any, float]:
        self._sync()
        start = time.perf_counter()
        out = fn(*args)
        self._sync()
        return out, (time.perf_counter() - start) * 1000

    def __call__(self, image: np.ndarray) -> Labels:
        (rows, labels), det_ms = self.timed(self.detect, image)
        h, w = image.shape[:2]
        keep, filter_ms = self.timed(filter_boxes, rows, self.config.box_filter, w, h)
        detected, rows, labels = len(rows), rows[keep], [labels[i] for i in keep]
        ms = {"detector": det_ms, "filter": filter_ms}
        masks = None
        if self.segment:
            masks, ms["segmenter"] = self.timed(self.segment, image, rows[:, :4])
        return Labels(rows, labels, masks, ms, detected)


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
