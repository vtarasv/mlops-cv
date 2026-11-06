"""The auto-labeler: SAM 3 or Grounding DINO on text prompts -> box filter, one RGB frame at a time,
then a box tracker over each clip.

Each stage is timed on its own, the GPU synchronised at its edges. Its auto-labels are written as
VisDrone-VID annotation lines, which ingestion reads like VisDrone's own.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from mlops_cv.config.settings import FLOOR_KEYS, AutolabelDetector, AutolabelTracker, BoxFilter
from mlops_cv.data.convert_visdrone_vid import CLASS_MERGE, RAW_NAMES, YOLO_NAMES

MERGED = {name: c for c, name in YOLO_NAMES.items()}
RAW_IDS = {name: c for c, name in RAW_NAMES.items()}
NO_TRACK = -1  # VisDrone-VID's target_id when no tracker ran

# rows and labels, one per detector box
Detect = Callable[[np.ndarray], tuple[np.ndarray, list[str]]]


@dataclass
class Labels:
    """One frame's boxes: the detector's, or the auto-labels the box filter keeps."""

    rows: np.ndarray  # (n, 6) x1 y1 x2 y2 conf merged-cls, source pixels
    labels: list[str]  # the detector's label per row (its prompt)
    ms: dict[str, float]  # wall-clock per stage
    detected: int  # detector boxes before the filter
    track_ids: np.ndarray | None = None  # the track per row, when a tracker ran
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


def above(labels: Labels, floor: float | None) -> Labels:
    """``labels`` without the rows scoring below ``floor``: the auto-labels that are kept."""
    if floor is None:
        return labels
    keep = np.flatnonzero(labels.rows[:, 4] >= floor)
    ids = None if labels.track_ids is None else labels.track_ids[keep]
    return replace(
        labels, rows=labels.rows[keep], labels=[labels.labels[i] for i in keep], track_ids=ids
    )


def torch_device(device: str) -> str:
    return f"cuda:{device}" if device.isdigit() else device


def top(conf: Any, cfg: AutolabelDetector, max_det: int) -> Any:
    """The positions of the top ``max_det`` scores above the floor (torch)."""
    best = conf.argsort(descending=True)[:max_det]
    return best[conf[best] > cfg.threshold]


def top_rows(
    boxes: Any, conf: Any, prompt: Any, cfg: AutolabelDetector, max_det: int
) -> tuple[np.ndarray, list[str]]:
    """The top ``max_det`` above the floor: xyxy pixel boxes, scores and prompt indices (torch)."""
    names = list(cfg.prompts)
    classes = np.array([MERGED[c] for c in cfg.prompts.values()])
    keep = top(conf, cfg, max_det)
    label = prompt[keep].cpu().numpy()
    rows = np.column_stack([boxes[keep].cpu().numpy(), conf[keep].cpu().numpy(), classes[label]])
    return rows.astype(np.float32).reshape(-1, 6), [names[i] for i in label]


def sam3(cfg: AutolabelDetector, device: str, max_det: int) -> Detect:
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
        return top_rows(torch.cat(boxes), torch.cat(conf).float(), torch.cat(label), cfg, max_det)

    return detect


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


def grounding_dino(cfg: AutolabelDetector, device: str, max_det: int) -> Detect:
    """Grounding DINO, each box to its top prompt: transformers' phrase decode can join two."""
    import torch
    from transformers import AutoConfig, AutoModelForZeroShotObjectDetection, AutoProcessor

    processor: Any = AutoProcessor.from_pretrained(cfg.model)
    # transformers 5.8 otherwise ties MM-GDINO's per-layer decoder heads to layer 0's
    share = getattr(AutoConfig.from_pretrained(cfg.model), "decoder_bbox_embed_share", True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        cfg.model, tie_word_embeddings=share
    )
    model = model.to(device).eval()
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


def detector(cfg: AutolabelDetector, device: str, max_det: int) -> Detect:
    """SAM 3 or Grounding DINO, picked by the model name."""
    return (sam3 if "sam3" in cfg.model else grounding_dino)(cfg, device, max_det)


class Labeler:
    """The detector and the box filter, loaded once; called on one decoded RGB image."""

    def __init__(
        self,
        detector: AutolabelDetector,
        box_filter: BoxFilter,
        device: str,
        max_det: int,
        detect: bool = True,
    ) -> None:
        self.detector, self.box_filter, self.max_det = detector, box_filter, max_det
        self.device = torch_device(device)
        self._sync: Callable[[], None] = lambda: None
        if self.device.startswith("cuda"):
            import torch

            self._sync = torch.cuda.synchronize
        if detect:
            self.detect = self.load()

    def load(self) -> Detect:
        return detector(self.detector, self.device, self.max_det)

    def timed(self, fn: Callable[..., Any], *args: Any) -> tuple[Any, float]:
        self._sync()
        start = time.perf_counter()
        out = fn(*args)
        self._sync()
        return out, (time.perf_counter() - start) * 1000

    def detections(self, image: np.ndarray) -> Labels:
        """The detector's boxes, before the box filter."""
        (rows, labels), ms = self.timed(self.detect, image)
        return Labels(rows, labels, {"detector": ms}, len(rows))

    def finish(self, image: np.ndarray, detections: Labels) -> Labels:
        """The box filter over one image's ``detections``."""
        h, w = image.shape[:2]
        rows, labels, ms = detections.rows, detections.labels, dict(detections.ms)
        keep, ms["filter"] = self.timed(filter_boxes, rows, self.box_filter, w, h)
        return Labels(rows[keep], [labels[i] for i in keep], ms, detections.detected)

    def __call__(self, image: np.ndarray) -> Labels:
        return self.finish(image, self.detections(image))


def tracker_args(cfg: AutolabelTracker, floor: float | None) -> dict[str, Any]:
    """``cfg.model``'s ultralytics YAML under ``cfg.overrides``; ``floor`` sets ``FLOOR_KEYS``."""
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    args = YAML.load(str(check_yaml(f"{cfg.model}.yaml")))
    if unknown := cfg.overrides.keys() - args.keys():
        raise ValueError(f"{cfg.model} has no {sorted(unknown)} in its YAML")
    args |= cfg.overrides
    if floor is not None:
        args |= dict.fromkeys(FLOOR_KEYS, floor)
    return args


def track_clip(
    model: str, args: dict[str, Any], clip: Iterable[tuple[np.ndarray, Labels]]
) -> Iterator[tuple[Labels, int]]:
    """One clip's (RGB frame, kept boxes) under a fresh tracker -> tracked boxes, live tracks."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.basetrack import BaseTrack
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import IterableSimpleNamespace

    tracker: Any = TRACKER_MAP[model](args=IterableSimpleNamespace(**args))
    BaseTrack.reset_id()  # TrackTrack has no reset_id of its own; the counter is shared
    for rgb, det in clip:
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # for camera-motion compensation
        start = time.perf_counter()
        rows = tracker.update(Boxes(det.rows, rgb.shape[:2]), bgr).reshape(-1, 8)
        ms = (time.perf_counter() - start) * 1000
        idx = rows[:, 7].astype(int)
        labels = Labels(
            det.rows[idx],
            [det.labels[i] for i in idx],
            det.ms | {"tracker": ms},
            det.detected,
            rows[:, 4].astype(int),
        )
        yield labels, len(tracker.tracked_stracks) + len(tracker.lost_stracks)


def label_clip(
    labeler: Labeler, tracker: AutolabelTracker, floor: float, frames: Iterable[np.ndarray]
) -> Iterator[str]:
    """One clip's RGB frames, from frame 1 on -> its VisDrone-VID annotation lines, tracked."""
    clip = ((rgb, labeler(rgb)) for rgb in frames)
    tracked = track_clip(tracker.model, tracker_args(tracker, floor), clip)
    for frame, (labels, _) in enumerate(tracked, start=1):
        yield from annotation_lines(frame, labels)


def raw_category(prompt: str, cls: int) -> int:
    """The raw VisDrone category a prompt names, else the lowest one merging into ``cls``."""
    return RAW_IDS.get(prompt) or min(c for c, m in CLASS_MERGE.items() if m == cls)


def annotation_lines(frame: int, labels: Labels) -> list[str]:
    """One frame's boxes as VisDrone-VID annotation lines, the score column their confidence."""
    ids = [NO_TRACK] * len(labels.rows) if labels.track_ids is None else labels.track_ids
    return [
        f"{frame},{int(t)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{conf:.6g},"
        f"{raw_category(prompt, int(cls))},-1,-1"
        for (x1, y1, x2, y2, conf, cls), prompt, t in zip(
            labels.rows, labels.labels, ids, strict=True
        )
    ]


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
