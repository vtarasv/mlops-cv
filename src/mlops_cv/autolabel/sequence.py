"""Sequence mode: key-frame tracking with a video model, and ultralytics' box-tracker controls.

A video trial runs the detector and box filter on key frames only. Kept boxes match the live
tracks by box IoU against each track's mask box; the rest join the running tracker as new objects.
The tracker is never reset, and its memory is pruned to the seed frames plus the last 16.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import groupby
from operator import attrgetter
from typing import Any

import numpy as np

from mlops_cv.autolabel import visdrone_vid
from mlops_cv.autolabel.labeler import (
    SEGMENT_CHUNK,
    Labeler,
    LabelingConfig,
    Labels,
    Tracker,
    read_rgb,
    rle,
)

logger = logging.getLogger(__name__)

# Non-seed frames a video tracker keeps: memory stays flat at unchanged latency (ticket 07).
KEEP_FRAMES = 16
# A transformers video session's per-object stores, indexed by object position.
PER_OBJECT = (
    "point_inputs_per_obj",
    "mask_inputs_per_obj",
    "output_dict_per_obj",
    "frames_tracked_per_obj",
)


@dataclass
class Track:
    label: str
    cls: int
    score: float  # the last matching detection's
    empty: int = 0  # consecutive empty-mask frames
    missed: int = 0  # consecutive key frames without a match


def match(iou: np.ndarray, tau: float) -> tuple[dict[int, int], np.ndarray]:
    """``iou`` (boxes, tracks) -> one-to-one ``{box: track}`` at IoU >= ``tau``, and the rest.

    The trackers' convention (SORT, ByteTrack, BoT-SORT): a linear assignment on 1 - IoU gated at
    ``tau``, through ultralytics' own; every box left unmatched starts a new track.
    """
    from ultralytics.trackers.utils.matching import linear_assignment

    matches, unmatched, _ = linear_assignment(1 - np.asarray(iou, float), 1 - tau)
    return {int(b): int(t) for b, t in matches}, np.asarray(unmatched, int)


def younger_duplicates(iou: np.ndarray, ids: list[int], threshold: float) -> set[int]:
    """Of each two tracks whose masks overlap at IoU > ``threshold``, the younger (higher id)."""
    ended: set[int] = set()
    for a, b in zip(*np.nonzero(np.triu(iou > threshold, 1)), strict=True):
        if ids[a] not in ended and ids[b] not in ended:
            ended.add(max(ids[a], ids[b]))
    return ended


class Tracks:
    """The key-frame rules' bookkeeping over the live tracks; ids count up, so higher is younger."""

    def __init__(self, cfg: Tracker) -> None:
        self.cfg = cfg
        self.live: dict[int, Track] = {}
        self._next = 1

    def _end(self, ids: set[int]) -> set[int]:
        for i in ids:
            del self.live[i]
        return ids

    def observe(self, ids: list[int], areas: Any) -> set[int]:
        """One frame's mask areas per track -> the tracks the empty-mask rule ends."""
        for i, area in zip(ids, areas, strict=True):
            track = self.live[i]
            track.empty = 0 if area else track.empty + 1
        return self._end({i for i in ids if self.live[i].empty >= self.cfg.empty_frames})

    def key_frame(
        self,
        ids: list[int],
        rows: np.ndarray,
        labels: list[str],
        box_iou: np.ndarray,
        mask_iou: np.ndarray,
    ) -> tuple[set[int], dict[int, int]]:
        """One key frame -> (the tracks it ends, ``{track: kept box}`` for matched and new tracks).

        ``rows`` are the filtered boxes, highest score first, so a cap keeps the best new ones.
        Duplicates end first, so a track about to end can't take a box.
        """
        cfg = self.cfg
        duplicates = younger_duplicates(mask_iou, ids, cfg.duplicate_iou)
        cols = [j for j, i in enumerate(ids) if i not in duplicates]
        pairs, new = match(box_iou[:, cols], cfg.new_iou)
        matched = {ids[cols[t]]: b for b, t in pairs.items()}
        for j in cols:
            track = self.live[ids[j]]
            if ids[j] in matched:
                track.missed, track.score = 0, float(rows[matched[ids[j]], 4])
            else:
                track.missed += 1
        drifted = {ids[j] for j in cols if self.live[ids[j]].missed >= cfg.unmatched_keys}
        ended = self._end(duplicates | drifted)
        room = len(new) if cfg.max_objects is None else max(0, cfg.max_objects - len(self.live))
        for b in new[:room]:
            self.live[self._next] = Track(labels[b], int(rows[b, 5]), float(rows[b, 4]))
            matched[self._next] = int(b)
            self._next += 1
        return ended, matched


def objects(session: Any) -> dict[int, tuple[Any, ...]]:
    """A video session's objects: ``{object id: its per-object stores}``, in position order."""
    return {
        oid: tuple(getattr(session, store)[i] for store in PER_OBJECT)
        for i, oid in enumerate(session.obj_ids)
    }


def set_objects(session: Any, state: dict[int, tuple[Any, ...]]) -> None:
    """Re-index ``session`` to hold exactly ``state``'s objects.

    ponytail: transformers 5.8.0 has no remove-object call, so this rewrites its private index
    maps; recheck it on a transformers bump.
    """
    session._obj_id_to_idx.clear()
    session._obj_idx_to_id.clear()
    for store in PER_OBJECT:
        getattr(session, store).clear()
    for i, (oid, parts) in enumerate(state.items()):
        session._obj_id_to_idx[oid] = i
        session._obj_idx_to_id[i] = oid
        for store, part in zip(PER_OBJECT, parts, strict=True):
            getattr(session, store)[i] = part
    session.obj_ids = list(state)
    session.obj_with_new_inputs = [o for o in session.obj_with_new_inputs if o in state]


def prune(session: Any, frame: int) -> None:
    """Keep each object's seed frame and its last ``KEEP_FRAMES`` tracked frames, drop the rest.

    ``high_res_masks`` is written for the memory encoder and never read back.
    """
    (session.processed_frames or {}).pop(frame, None)
    for out in session.output_dict_per_obj.values():
        recent = out["non_cond_frame_outputs"]
        for t in [t for t in recent if t < frame - KEEP_FRAMES]:
            del recent[t]
        for store in (recent, out["cond_frame_outputs"]):
            if frame in store:
                store[frame].pop("high_res_masks", None)


def coco_iou(a: Any, b: Any) -> np.ndarray:
    """pycocotools' IoU between xywh boxes or RLEs, shaped ``(len(a), len(b))`` even when empty."""
    from pycocotools import mask

    mask_utils: Any = mask
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    return np.asarray(mask_utils.iou(a, b, [0] * len(b)), float)


class VideoTracker:
    """A transformers SAM 2-family video model (EdgeTAM, SAM 2.1) under the key-frame rules."""

    def __init__(self, cfg: Tracker, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.cfg, self.device, self.dtype = cfg, device, torch.bfloat16
        self.processor: Any = AutoProcessor.from_pretrained(cfg.model)
        model: Any = AutoModel.from_pretrained(cfg.model)
        self.model = model.to(device, dtype=self.dtype).eval()

    def _forward(self, session: Any, inputs: Any, k: int) -> tuple[list[int], list[dict]]:
        out = self.model(inference_session=session, frame=inputs.pixel_values[0], frame_idx=k)
        rles: list[dict] = []
        for c in range(0, len(out.pred_masks), SEGMENT_CHUNK):
            chunk = [out.pred_masks[c : c + SEGMENT_CHUNK]]
            masks = self.processor.post_process_masks(chunk, inputs.original_sizes, binarize=True)
            rles += rle(masks[0][:, 0])
        return list(out.object_ids), rles

    def _add(self, session: Any, inputs: Any, k: int, ids: list[int], boxes: np.ndarray) -> list:
        """Seed new objects on frame ``k`` with the live ones hidden, so they don't run twice."""
        live = objects(session)
        set_objects(session, {})
        self.processor.add_inputs_to_inference_session(
            session,
            frame_idx=k,
            obj_ids=list(ids),  # the session keeps this list and empties it as objects run
            input_boxes=[boxes.tolist()],
            original_size=inputs.original_sizes[0],
        )
        _, rles = self._forward(session, inputs, k)
        set_objects(session, live | objects(session))
        return rles

    def _step(
        self, session: Any, tracks: Tracks, rgb: np.ndarray, k: int, key: Labels | None
    ) -> tuple[dict[int, dict], dict[int, int]]:
        """Track frame ``k``, then a key frame's rules -> live masks, ``{track: kept box}``."""
        from pycocotools import mask

        mask_utils: Any = mask
        inputs = self.processor(images=rgb, device=self.device, return_tensors="pt")
        ids, rles = self._forward(session, inputs, k) if tracks.live else ([], [])
        ended = tracks.observe(ids, mask_utils.area(rles) if rles else [])
        masks = {i: m for i, m in zip(ids, rles, strict=True) if i not in ended}
        boxes: dict[int, int] = {}
        if key is not None:
            ids = list(masks)
            xywh = np.column_stack([key.rows[:, :2], key.rows[:, 2:4] - key.rows[:, :2]])
            mask_boxes = mask_utils.toBbox(list(masks.values())) if masks else []
            key_ended, boxes = tracks.key_frame(
                ids,
                key.rows,
                key.labels,
                coco_iou(xywh.astype(float), mask_boxes),
                coco_iou(list(masks.values()), list(masks.values())),
            )
            ended |= key_ended
            new = [i for i in boxes if i not in masks]
            if new:
                seeds = key.rows[[boxes[i] for i in new], :4]
                masks |= dict(zip(new, self._add(session, inputs, k, new, seeds), strict=True))
        if ended:
            set_objects(session, {o: s for o, s in objects(session).items() if o not in ended})
            masks = {i: m for i, m in masks.items() if i not in ended}
        prune(session, k)
        return masks, boxes

    def frames(
        self, labeler: Labeler, images: list[visdrone_vid.Image]
    ) -> Iterator[tuple[Labels, Labels | None, int]]:
        """One sequence -> per frame: its auto-labels, the key frame's detections, live tracks.

        An empty mask is no auto-label. A row's box is its matching detection's on a key frame
        and NaN elsewhere: the auto-label's box is its mask's.
        """
        import torch
        from pycocotools import mask

        mask_utils: Any = mask
        session = self.processor.init_video_session(
            inference_device=self.device, video_storage_device="cpu", dtype=self.dtype
        )
        tracks = Tracks(self.cfg)
        for k, im in enumerate(images):
            rgb = read_rgb(im.path)
            key = labeler(rgb) if k % self.cfg.key_interval == 0 else None
            with torch.inference_mode():
                (masks, boxes), ms = labeler.timed(self._step, session, tracks, rgb, k, key)
            ids = [i for i, m in masks.items() if mask_utils.area(m)]
            rows = np.full((len(ids), 6), np.nan, np.float32)
            for row, i in zip(rows, ids, strict=True):
                if key is not None and i in boxes:
                    row[:4] = key.rows[boxes[i], :4]
                row[4:] = tracks.live[i].score, tracks.live[i].cls
            labels = Labels(
                rows,
                [tracks.live[i].label for i in ids],
                [masks[i] for i in ids],
                (key.ms if key else {}) | {"tracker": ms},
                key.detected if key else 0,
                np.array(ids, int),
            )
            yield labels, key, len(tracks.live)


def video_tracks(
    config: LabelingConfig, images: list[visdrone_vid.Image], device: str
) -> Iterator[tuple[Labels, Labels | None, int]]:
    """A video-tracker trial over every sequence in ``images``, a new session per sequence."""
    assert config.tracker is not None
    labeler = Labeler(config, device)
    tracker = VideoTracker(config.tracker, labeler.device)
    for sequence, frames in groupby(images, key=attrgetter("sequence")):
        frames = list(frames)
        peak = 0
        for out in tracker.frames(labeler, frames):
            peak = max(peak, out[2])
            yield out
        logger.info(f"  {sequence}: {len(frames)} frames, peak {peak} live tracks")
    del labeler, tracker
    gc.collect()


def box_tracks(
    name: str, images: list[visdrone_vid.Image], detections: list[Labels]
) -> Iterator[tuple[Labels, int]]:
    """ultralytics' ``name`` tracker at its defaults over a cached detection pass, per frame.

    The tracker reports the detections it assigned a track to; their rows and masks are kept.
    """
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    args = IterableSimpleNamespace(**YAML.load(str(check_yaml(f"{name}.yaml"))))
    tracker: Any = None
    sequence = None
    for im, det in zip(images, detections, strict=True):
        if im.sequence != sequence:
            sequence, tracker = im.sequence, TRACKER_MAP[name](args=args)
            tracker.reset_id()
        bgr = np.ascontiguousarray(read_rgb(im.path)[..., ::-1])  # BoT-SORT's motion compensation
        start = time.perf_counter()
        out = tracker.update(Boxes(det.rows, (im.height, im.width)), bgr).reshape(-1, 8)
        ms = (time.perf_counter() - start) * 1000
        idx = out[:, 7].astype(int)
        masks = None if det.masks is None else [det.masks[i] for i in idx]
        labels = Labels(
            det.rows[idx],
            [det.labels[i] for i in idx],
            masks,
            det.ms | {"tracker": ms},
            det.detected,
            out[:, 4].astype(int),
        )
        yield labels, len(tracker.tracked_stracks) + len(tracker.lost_stracks)
