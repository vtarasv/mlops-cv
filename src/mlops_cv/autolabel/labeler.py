"""The labeling config and the auto-labeler it builds: one decoded image -> boxes and RLE masks.

Every model loads through transformers, except YOLOE and the champion's YOLO26 (ultralytics).
Each stage is timed on its own, the GPU synchronised at its edges.
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
from mlops_cv.evaluation.score_detections import CONF, MAX_DET

MergedClass = Literal["person", "vehicle", "two-three-wheeler"]
MERGED = {name: c for c, name in YOLO_NAMES.items()}

# The champion's YOLO26, resolved from the registry: the supervised reference row.
CHAMPION = "champion"
# Boxes per SAM 2 mask-decoder pass: bounds the full-resolution mask upsample's memory.
SEGMENT_CHUNK = 32
# ultralytics' box trackers, run with their default settings; any other tracker is a video model.
BOX_TRACKERS = ("botsort", "bytetrack")
# Tiling's fixed settings: SAHI's slice overlap, the cut band at an interior slice edge (px), and
# the class-aware NMS over all passes.
OVERLAP, EDGE_MARGIN, TILE_NMS_IOU = 0.2, 2, 0.5
# The comparator merge: a same-class box this share inside a larger one is a cut-off piece of it.
CONTAINED_IOS = 0.85

Detect = Callable[[np.ndarray], tuple[np.ndarray, list[str]]]
Segment = Callable[[np.ndarray, np.ndarray], list[dict[str, Any]]]


class Tiling(BaseModel):
    """Slices plus the full frame, merged before the box filter."""

    model_config = ConfigDict(extra="forbid")

    side: int  # slice side in px; zoom is the detector's input side over it (SAM 3: 1008)
    # ``edge`` drops boxes cut by an interior slice edge; ``contained`` merges a cut-off piece
    # into the larger box around it. Both then run class-aware NMS.
    merge: Literal["edge", "contained"] = "edge"


class Detector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str  # a transformers repo id, an ultralytics YOLOE weights file, or ``champion``
    prompts: dict[str, MergedClass] = {}  # prompt text -> the merged class its boxes get
    threshold: float = CONF  # the detector's own score floor; it keeps its top ``MAX_DET``
    imgsz: int | None = None  # YOLOE's input size; ``None`` is 640, its training size
    tiling: Tiling | None = None
    dtype: Literal["bfloat16"] | None = None  # SAM 3's weights; ``None`` is fp32


class BoxFilter(BaseModel):
    """The box filter's stages, applied in this order; ``None`` turns a stage off."""

    model_config = ConfigDict(extra="forbid")

    min_side: float | None = None  # px
    max_area: float | None = None  # share of the frame
    min_score: float | None = None
    nms_iou: float | None = None  # class-agnostic
    nested_ios: float | None = None  # of two boxes this share nested, drop one
    nested_keep: Literal["outer", "inner"] = "outer"  # ... the box the nested-box rule keeps
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
    # The winning trial's max-F1 confidence: the stored auto-labels are cut there, scoring isn't.
    confidence_floor: float | None = None

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
    windows: np.ndarray | None = None  # tiled, before the merge: each row's pass, xyxy px


def ios(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Intersection over the smaller box's area, ``box`` (4,) against ``others`` (m, 4) xyxy."""
    lt = np.maximum(box[:2], others[:, :2])
    rb = np.minimum(box[2:], others[:, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    own = (box[2] - box[0]) * (box[3] - box[1])
    return inter / (np.minimum(own, area) + 1e-9)


def drop_nested(boxes: np.ndarray, threshold: float, keep_outer: bool = True) -> np.ndarray:
    """The nested-box rule: of two boxes nested at IoS >= ``threshold``, the outer or inner goes.

    Keeping the outer box, a detector box drawn around a group loses the objects inside it;
    keeping the inner, the group box goes. Returns the kept positions.
    """
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    for i in np.argsort(-area if keep_outer else area, kind="stable"):
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
        idx = idx[drop_nested(rows[idx, :4], f.nested_ios, f.nested_keep == "outer")]
        idx = idx[np.argsort(-conf[idx], kind="stable")]
    return idx[: f.max_objects]


def _starts(n: int, side: int, step: int) -> list[int]:
    return [0] if n <= side else [*range(0, n - side, step), n - side]


def slice_grid(width: int, height: int, side: int) -> list[tuple[int, int, int, int]]:
    """SAHI's ``get_slice_bboxes`` at ``OVERLAP``: xyxy per slice, the last row and column flush."""
    step = side - int(OVERLAP * side)
    return [
        (x, y, min(x + side, width), min(y + side, height))
        for y in _starts(height, side, step)
        for x in _starts(width, side, step)
    ]


def tiled(detect: Detect, image: np.ndarray, side: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    """The detector on each slice, then on the full frame: rows in frame px, labels, windows.

    A frame that fits one slice is a single pass.
    """
    h, w = image.shape[:2]
    windows = slice_grid(w, h, side)
    if len(windows) > 1:
        windows.append((0, 0, w, h))
    rows, labels, wins = [], [], []
    for x0, y0, x1, y1 in windows:
        r, lab = detect(image[y0:y1, x0:x1])
        rows.append(r + np.array([x0, y0, x0, y0, 0, 0], np.float32))
        labels += lab
        wins.append(np.tile(np.array([x0, y0, x1, y1], np.float32), (len(r), 1)))
    return np.concatenate(rows), labels, np.concatenate(wins)


def cut_by_slice_edge(
    boxes: np.ndarray, windows: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Boxes within ``EDGE_MARGIN`` of their slice's interior edges, which cut objects."""
    x0, y0, x1, y1 = windows.T
    m = EDGE_MARGIN
    return (
        ((x0 > 0) & (boxes[:, 0] <= x0 + m))
        | ((y0 > 0) & (boxes[:, 1] <= y0 + m))
        | ((x1 < width) & (boxes[:, 2] >= x1 - m))
        | ((y1 < height) & (boxes[:, 3] >= y1 - m))
    )


def merge_contained(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Larger box wins: a box ``CONTAINED_IOS`` inside a larger kept one of its class joins it.

    The kept box takes the pair's top score. Returns the kept positions and their scores.
    """
    area = (rows[:, 2] - rows[:, 0]) * (rows[:, 3] - rows[:, 1])
    conf = rows[:, 4].copy()
    keep: list[int] = []
    for i in np.argsort(-area, kind="stable"):
        if keep:
            k = np.array(keep)
            hit = k[(ios(rows[i, :4], rows[k, :4]) >= CONTAINED_IOS) & (rows[k, 5] == rows[i, 5])]
            if len(hit):
                conf[hit[0]] = max(conf[hit[0]], conf[i])
                continue
        keep.append(int(i))
    kept = np.array(keep, int)
    return kept, conf[kept]


def merge_passes(
    rows: np.ndarray, windows: np.ndarray, merge: str, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """A tiled frame's merge: the edge rule or the contained merge, with class-aware NMS.

    Returns the kept positions and their scores.
    """
    import torch
    from torchvision.ops import batched_nms

    idx = np.arange(len(rows))
    if merge == "edge":
        idx = idx[~cut_by_slice_edge(rows[:, :4], windows, width, height)]
    t = torch.from_numpy(rows[idx])
    idx = idx[batched_nms(t[:, :4], t[:, 4], t[:, 5].long(), TILE_NMS_IOU).numpy()]
    conf = rows[idx, 4]
    if merge == "contained":
        kept, conf = merge_contained(rows[idx])
        idx = idx[kept]
    return idx, conf


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


def top_rows(boxes: Any, conf: Any, prompt: Any, cfg: Detector) -> tuple[np.ndarray, list[str]]:
    """The top ``MAX_DET`` above the floor: xyxy pixel boxes, scores and prompt indices (torch)."""
    names = list(cfg.prompts)
    classes = np.array([MERGED[c] for c in cfg.prompts.values()])
    top = conf.argsort(descending=True)[:MAX_DET]
    keep = top[conf[top] > cfg.threshold]
    label = prompt[keep].cpu().numpy()
    rows = np.column_stack([boxes[keep].cpu().numpy(), conf[keep].cpu().numpy(), classes[label]])
    return rows.astype(np.float32).reshape(-1, 6), [names[i] for i in label]


def grounding_dino(cfg: Detector, device: str) -> Detect:
    """Grounding DINO: each box goes to the prompt whose tokens score it highest.

    transformers' own post-processing decodes the tokens above a threshold into a phrase, which
    can run two prompts together; the per-prompt maximum can't.
    """
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor: Any = AutoProcessor.from_pretrained(cfg.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.model).to(device).eval()
    text = ". ".join(p.strip().lower() for p in cfg.prompts) + "."
    tok = processor.tokenizer
    separators = {tok.cls_token_id, tok.sep_token_id, tok.convert_tokens_to_ids(".")}

    def detect(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        h, w = image.shape[:2]
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inputs)
        spans = prompt_spans(inputs["input_ids"][0].tolist(), separators)
        if len(spans) != len(cfg.prompts):
            raise ValueError(f"{len(spans)} token spans for {len(cfg.prompts)} prompts in {text!r}")
        probs = out.logits[0].sigmoid()
        conf, label = torch.stack([probs[:, s].max(-1).values for s in spans], 1).max(1)
        cx, cy, bw, bh = (out.pred_boxes[0] * torch.tensor([w, h, w, h], device=device)).T
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        return top_rows(boxes, conf, label, cfg)

    return detect


def owlv2(cfg: Detector, device: str) -> Detect:
    """OWLv2: one box per patch, labelled with its best-scoring prompt."""
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    processor: Any = Owlv2Processor.from_pretrained(cfg.model)
    model: Any = Owlv2ForObjectDetection.from_pretrained(cfg.model)
    model = model.to(device).eval()
    text = [list(cfg.prompts)]

    def detect(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        side = max(image.shape[:2])  # boxes are relative to the image padded to a square
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inputs)
        conf, label = out.logits[0].sigmoid().max(-1)
        cx, cy, bw, bh = (out.pred_boxes[0] * side).T
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        return top_rows(boxes, conf, label, cfg)

    return detect


def sam3(cfg: Detector, device: str) -> Detect:
    """SAM 3 end to end: one vision pass, then a decoder pass per prompt (200 boxes each)."""
    import torch
    from transformers import Sam3Model, Sam3Processor

    dtype = getattr(torch, cfg.dtype or "float32")
    processor: Any = Sam3Processor.from_pretrained(cfg.model)
    model: Any = Sam3Model.from_pretrained(cfg.model, dtype=dtype)
    model = model.to(device).eval()
    texts = [processor(text=p, return_tensors="pt").to(device) for p in cfg.prompts]

    def detect(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        h, w = image.shape[:2]
        inputs = processor(images=image, return_tensors="pt").to(device)
        boxes, conf, label = [], [], []
        with torch.inference_mode():
            vision = model.get_vision_features(pixel_values=inputs["pixel_values"].to(dtype))
            for i, text in enumerate(texts):
                out = model(vision_embeds=vision, **text)
                boxes.append(out.pred_boxes[0].float() * torch.tensor([w, h, w, h], device=device))
                conf.append(out.pred_logits[0].sigmoid() * out.presence_logits[0].sigmoid())
                label.append(torch.full_like(conf[-1], i, dtype=torch.long))
        return top_rows(torch.cat(boxes), torch.cat(conf).float(), torch.cat(label), cfg)

    return detect


def yoloe(cfg: Detector, device: str) -> Detect:
    """YOLOE text-prompted, with val's floor, cap and one-to-many head, like the champion."""
    from ultralytics import YOLOE

    model = YOLOE(cfg.model)
    model.set_classes(list(cfg.prompts))

    def detect(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        bgr = np.ascontiguousarray(image[..., ::-1])
        result: Any = model.predict(
            bgr,
            imgsz=cfg.imgsz or 640,
            device=device,
            conf=cfg.threshold,
            max_det=MAX_DET,
            end2end=False,
            verbose=False,
        )[0]
        data = result.boxes.data
        return top_rows(data[:, :4], data[:, 4], data[:, 5].long(), cfg)

    return detect


def detector(cfg: Detector, device: str) -> Detect:
    """The detector's loader, picked by its model name."""
    if cfg.model == CHAMPION:
        return champion(device)
    loaders = {"yoloe": yoloe, "owlv2": owlv2, "sam3": sam3}
    load = next((f for key, f in loaders.items() if key in cfg.model), grounding_dino)
    return load(cfg, device)


def champion(device: str) -> Detect:
    """The champion YOLO26 at its training imgsz, with val's floor, cap and one-to-many head."""
    from ultralytics import YOLO

    from mlops_cv.config import get_settings
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

    def __init__(self, config: LabelingConfig, device: str, detect: bool = True) -> None:
        self.config = config
        self.device = device = _device(device)
        self._sync: Callable[[], None] = lambda: None
        if device.startswith("cuda"):
            import torch

            self._sync = torch.cuda.synchronize
        if detect:
            self.detect = detector(config.detector, device)
        self.segment = sam2(config.segmenter, device) if config.segmenter else None

    def timed(self, fn: Callable[..., Any], *args: Any) -> tuple[Any, float]:
        self._sync()
        start = time.perf_counter()
        out = fn(*args)
        self._sync()
        return out, (time.perf_counter() - start) * 1000

    def detections(self, image: np.ndarray) -> Labels:
        """The detector's boxes, before the box filter; tiled, every pass's before the merge."""
        tiling = self.config.detector.tiling
        if tiling is None:
            (rows, labels), ms = self.timed(self.detect, image)
            return Labels(rows, labels, None, {"detector": ms}, len(rows))
        (rows, labels, windows), ms = self.timed(tiled, self.detect, image, tiling.side)
        return Labels(rows, labels, None, {"detector": ms}, len(rows), windows=windows)

    def finish(self, image: np.ndarray, detections: Labels) -> Labels:
        """The tiles' merge, the box filter, then the segmenter, over one image's ``detections``."""
        h, w = image.shape[:2]
        rows, labels, ms = detections.rows, detections.labels, dict(detections.ms)
        tiling = self.config.detector.tiling
        if tiling and detections.windows is not None:
            args = (rows, detections.windows, tiling.merge, w, h)
            (keep, conf), ms["merge"] = self.timed(merge_passes, *args)
            rows, labels = rows[keep], [labels[i] for i in keep]
            rows[:, 4] = conf
        keep, ms["filter"] = self.timed(filter_boxes, rows, self.config.box_filter, w, h)
        rows, labels = rows[keep], [labels[i] for i in keep]
        masks = None
        if self.segment:
            masks, ms["segmenter"] = self.timed(self.segment, image, rows[:, :4])
        return Labels(rows, labels, masks, ms, detections.detected)

    def __call__(self, image: np.ndarray) -> Labels:
        return self.finish(image, self.detections(image))


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
