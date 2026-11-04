"""The labeling trials' detectors beyond production's SAM 3: GDINO, OWLv2, YOLOE, and tiling.

Every model loads through transformers, except YOLOE (ultralytics).
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from mlops_cv.autolabel.labeler import Detect, Labeler, Labels, ios, sam3, top_rows
from mlops_cv.config.settings import AutolabelDetector

# Tiling's fixed settings: SAHI's slice overlap, the cut band at an interior slice edge (px), and
# the class-aware NMS over all passes.
OVERLAP, EDGE_MARGIN, TILE_NMS_IOU = 0.2, 2, 0.5
# The comparator merge: a same-class box this share inside a larger one is a cut-off piece of it.
CONTAINED_IOS = 0.85


class Tiling(BaseModel):
    """Slices plus the full frame, merged before the box filter."""

    model_config = ConfigDict(extra="forbid")

    side: int  # slice side in px; zoom is the detector's input side over it (SAM 3: 1008)
    # ``edge`` drops boxes cut by an interior slice edge; ``contained`` merges a cut-off piece
    # into the larger box around it. Both then run class-aware NMS.
    merge: Literal["edge", "contained"] = "edge"
    # ``False`` leaves out the full-frame pass, which a Swin-L GDINO can't fit in 8 GB
    full_frame: bool | None = None


class Detector(AutolabelDetector):
    """Production's detector settings plus the knobs only labeling trials turn."""

    imgsz: int | None = None  # YOLOE's input size; ``None`` is 640, its training size
    tiling: Tiling | None = None


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


def tiled(
    detect: Detect, image: np.ndarray, side: int, full_frame: bool = True
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """The detector on each slice, then on the full frame: rows in frame px, labels, windows.

    A frame that fits one slice is a single pass.
    """
    h, w = image.shape[:2]
    windows = slice_grid(w, h, side)
    if len(windows) > 1 and full_frame:
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


def merge_contained(rows: np.ndarray, cut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Larger box wins: a ``cut`` box ``CONTAINED_IOS`` inside a larger kept one of its class joins.

    Only a box cut by its slice's edge is a piece; a whole object inside a group box stays. The
    kept box takes the pair's top score. Returns the kept positions and their scores.
    """
    area = (rows[:, 2] - rows[:, 0]) * (rows[:, 3] - rows[:, 1])
    conf = rows[:, 4].copy()
    keep: list[int] = []
    for i in np.argsort(-area, kind="stable"):
        if keep and cut[i]:
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

    cut = cut_by_slice_edge(rows[:, :4], windows, width, height)
    idx = np.arange(len(rows))
    if merge == "edge":
        idx = idx[~cut]
    t = torch.from_numpy(rows[idx])
    idx = idx[batched_nms(t[:, :4], t[:, 4], t[:, 5].long(), TILE_NMS_IOU).numpy()]
    conf = rows[idx, 4]
    if merge == "contained":
        kept, conf = merge_contained(rows[idx], cut[idx])
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


def autocast(dtype: str | None, device: str) -> AbstractContextManager[Any]:
    """``dtype`` through autocast, not half weights: transformers' GDINO keeps some tensors fp32."""
    import torch

    return (
        torch.autocast(torch.device(device).type, getattr(torch, dtype)) if dtype else nullcontext()
    )


def grounding_dino(cfg: Detector, device: str, max_det: int) -> Detect:
    """Grounding DINO: each box goes to the prompt whose tokens score it highest.

    transformers' own post-processing decodes the tokens above a threshold into a phrase, which
    can run two prompts together; the per-prompt maximum can't.
    """
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor: Any = AutoProcessor.from_pretrained(cfg.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.model).to(device).eval()
    precision = autocast(cfg.dtype, device)
    text = ". ".join(p.strip().lower() for p in cfg.prompts) + "."
    tok = processor.tokenizer
    separators = {tok.cls_token_id, tok.sep_token_id, tok.convert_tokens_to_ids(".")}

    def detect(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        h, w = image.shape[:2]
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.inference_mode(), precision:
            out = model(**inputs)
        spans = prompt_spans(inputs["input_ids"][0].tolist(), separators)
        if len(spans) != len(cfg.prompts):
            raise ValueError(f"{len(spans)} token spans for {len(cfg.prompts)} prompts in {text!r}")
        probs = out.logits[0].float().sigmoid()
        conf, label = torch.stack([probs[:, s].max(-1).values for s in spans], 1).max(1)
        cx, cy, bw, bh = (out.pred_boxes[0].float() * torch.tensor([w, h, w, h], device=device)).T
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        return top_rows(boxes, conf, label, cfg, max_det)

    return detect


def owlv2(cfg: Detector, device: str, max_det: int) -> Detect:
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
        return top_rows(boxes, conf, label, cfg, max_det)

    return detect


def yoloe(cfg: Detector, device: str, max_det: int) -> Detect:
    """YOLOE text-prompted, with val's floor, cap and one-to-many head."""
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
            max_det=max_det,
            end2end=False,
            verbose=False,
        )[0]
        data = result.boxes.data
        return top_rows(data[:, :4], data[:, 4], data[:, 5].long(), cfg, max_det)

    return detect


def detector(cfg: Detector, device: str, max_det: int) -> Detect:
    """The detector's loader, picked by its model name."""
    loaders = {"yoloe": yoloe, "owlv2": owlv2, "sam3": sam3}
    load = next((f for key, f in loaders.items() if key in cfg.model), grounding_dino)
    return load(cfg, device, max_det)


class TrialLabeler(Labeler):
    """The production labeler with any trial detector, tiled or not."""

    detector: Detector

    def load(self) -> Detect:
        return detector(self.detector, self.device, self.max_det)

    def detections(self, image: np.ndarray) -> Labels:
        """The detector's boxes, before the box filter; tiled, every pass's before the merge."""
        tiling = self.detector.tiling
        if tiling is None:
            return super().detections(image)
        full = tiling.full_frame is not False
        (rows, labels, windows), ms = self.timed(tiled, self.detect, image, tiling.side, full)
        return Labels(rows, labels, {"detector": ms}, len(rows), windows=windows)

    def finish(self, image: np.ndarray, detections: Labels) -> Labels:
        """The tiles' merge, then the box filter."""
        tiling = self.detector.tiling
        if tiling and detections.windows is not None:
            h, w = image.shape[:2]
            rows, labels = detections.rows, detections.labels
            args = (rows, detections.windows, tiling.merge, w, h)
            (keep, conf), ms = self.timed(merge_passes, *args)
            rows = rows[keep]
            rows[:, 4] = conf
            detections = Labels(
                rows,
                [labels[i] for i in keep],
                detections.ms | {"merge": ms},
                detections.detected,
            )
        return super().finish(image, detections)
