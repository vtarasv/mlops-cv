"""Tracking for labeling trials: ultralytics' box trackers, and video trackers on key frames.

A video trial runs the detector (or reads its stored boxes) and the box filter on key frames
only. Kept boxes at or above the confidence floor match the live tracks by box IoU against each
track's box; the rest join the running tracker as new objects. A video tracker's masks never
leave it: each is cut to its tight box. The tracker is never reset, and its memory is pruned to
the seed frames plus the last 16.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import groupby
from operator import attrgetter
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from mlops_cv.autolabel.labeler import Labeler, Labels, above, read_rgb
from mlops_cv.tuning.autolabel_visdrone_vid import visdrone_vid

logger = logging.getLogger(__name__)

# ultralytics' box trackers; any other tracker is a video model.
BOX_TRACKERS = ("botsort", "bytetrack", "tracktrack")
# Box-tracker processes, one clip at a time each.
WORKERS = 16
# The box-tracker keys the confidence floor sets.
FLOOR_KEYS = frozenset({"track_high_thresh", "new_track_thresh"})
# Non-seed frames a video tracker keeps: memory stays flat at unchanged latency.
KEEP_FRAMES = 16
# Masks per post-processing pass: bounds the full-resolution masks' memory.
MASK_CHUNK = 32
# A transformers video session's per-object stores, indexed by object position.
PER_OBJECT = (
    "point_inputs_per_obj",
    "mask_inputs_per_obj",
    "output_dict_per_obj",
    "frames_tracked_per_obj",
)


class Tracker(BaseModel):
    """The key-frame rules apply to the video trackers, not to ``BOX_TRACKERS``."""

    model_config = ConfigDict(extra="forbid")

    model: str  # one of ``BOX_TRACKERS``, or a transformers video repo (EdgeTAM, SAM 2.1)
    key_interval: int = 10  # K: frame 0 and every K-th frame after it are key frames
    new_iou: float = 0.3  # tau: a kept box below it against every live track is a new object
    empty_frames: int = 25  # a track ends after this many consecutive empty-mask frames
    unmatched_keys: int = 2  # ... or after this many consecutive key frames without a match
    duplicate_iou: float = 0.8  # of two tracks whose boxes overlap above it, the younger ends
    max_objects: int | None = None  # live-track cap; new objects beyond it aren't added
    overrides: dict[str, Any] = {}  # a box tracker's ultralytics YAML keys over its defaults

    @model_validator(mode="after")
    def _overrides_need_a_box_tracker(self) -> Tracker:
        if self.overrides and self.model not in BOX_TRACKERS:
            raise ValueError("overrides apply only to a box tracker")
        return self


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
    """Of each two tracks whose boxes overlap at IoU > ``threshold``, the younger (higher id)."""
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
        """One frame's box areas per track -> the tracks the empty-mask rule ends."""
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
        track_iou: np.ndarray,
    ) -> tuple[set[int], dict[int, int]]:
        """One key frame -> (the tracks it ends, ``{track: kept box}`` for matched and new tracks).

        ``rows`` are the filtered boxes, highest score first, so a cap keeps the best new ones.
        Duplicates end first, so a track about to end can't take a box.
        """
        cfg = self.cfg
        duplicates = younger_duplicates(track_iou, ids, cfg.duplicate_iou)
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


def tight_boxes(masks: Any) -> np.ndarray:
    """``(n, H, W)`` bool torch masks -> ``(n, 4)`` xyxy tight boxes; an empty mask's is all 0."""
    import torch

    cols, rows = masks.any(1), masks.any(2)
    x1, y1 = cols.int().argmax(1), rows.int().argmax(1)
    x2 = cols.shape[1] - cols.flip(1).int().argmax(1)
    y2 = rows.shape[1] - rows.flip(1).int().argmax(1)
    boxes = torch.stack([x1, y1, x2, y2], 1) * cols.any(1, keepdim=True)
    return boxes.float().cpu().numpy().reshape(-1, 4)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy boxes, shaped ``(len(a), len(b))`` even when empty."""
    import torch
    from torchvision.ops import box_iou as iou

    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    return iou(torch.as_tensor(a, dtype=torch.float), torch.as_tensor(b, dtype=torch.float)).numpy()


def area(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


class VideoTracker:
    """A transformers SAM 2-family video model (EdgeTAM, SAM 2.1) under the key-frame rules."""

    def __init__(self, cfg: Tracker, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.cfg, self.device, self.dtype = cfg, device, torch.bfloat16
        self.processor: Any = AutoProcessor.from_pretrained(cfg.model)
        model: Any = AutoModel.from_pretrained(cfg.model)
        self.model = model.to(device, dtype=self.dtype).eval()

    def _forward(self, session: Any, inputs: Any, k: int) -> tuple[list[int], np.ndarray]:
        """One frame through the session -> its object ids and their masks' tight boxes."""
        out = self.model(inference_session=session, frame=inputs.pixel_values[0], frame_idx=k)
        boxes = [np.zeros((0, 4), np.float32)]
        for c in range(0, len(out.pred_masks), MASK_CHUNK):
            chunk = [out.pred_masks[c : c + MASK_CHUNK]]
            masks = self.processor.post_process_masks(chunk, inputs.original_sizes, binarize=True)
            boxes.append(tight_boxes(masks[0][:, 0]))
        return list(out.object_ids), np.concatenate(boxes)

    def _add(
        self, session: Any, inputs: Any, k: int, ids: list[int], boxes: np.ndarray
    ) -> np.ndarray:
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
        _, tracked = self._forward(session, inputs, k)
        set_objects(session, live | objects(session))
        return tracked

    def _step(
        self, session: Any, tracks: Tracks, rgb: np.ndarray, k: int, key: Labels | None
    ) -> dict[int, np.ndarray]:
        """Track frame ``k``, then a key frame's rules -> each live track's box."""
        inputs = self.processor(images=rgb, device=self.device, return_tensors="pt")
        ids, tracked = self._forward(session, inputs, k) if tracks.live else ([], np.zeros((0, 4)))
        ended = tracks.observe(ids, area(tracked))
        boxes = {i: b for i, b in zip(ids, tracked, strict=True) if i not in ended}
        if key is not None:
            ids, live = list(boxes), np.array(list(boxes.values())).reshape(-1, 4)
            key_ended, matched = tracks.key_frame(
                ids, key.rows, key.labels, box_iou(key.rows[:, :4], live), box_iou(live, live)
            )
            ended |= key_ended
            new = [i for i in matched if i not in boxes]
            if new:
                seeds = key.rows[[matched[i] for i in new], :4]
                boxes |= dict(zip(new, self._add(session, inputs, k, new, seeds), strict=True))
        if ended:
            set_objects(session, {o: s for o, s in objects(session).items() if o not in ended})
            boxes = {i: b for i, b in boxes.items() if i not in ended}
        prune(session, k)
        return boxes

    def frames(
        self,
        labeler: Labeler,
        images: list[visdrone_vid.Image],
        stored: dict[str, Labels] | None = None,
        floor: float | None = None,
    ) -> Iterator[tuple[Labels, Labels | None, int]]:
        """One sequence -> per frame: its auto-labels, the key frame's detections, live tracks.

        ``stored`` detector boxes, by frame name, replace the detector. Only kept boxes at or
        above ``floor`` start or refresh a track. An empty mask is no auto-label.
        """
        import torch

        session = self.processor.init_video_session(
            inference_device=self.device, video_storage_device="cpu", dtype=self.dtype
        )
        tracks = Tracks(self.cfg)
        for k, im in enumerate(images):
            rgb = read_rgb(im.path)
            key = None
            if k % self.cfg.key_interval == 0:
                key = labeler.finish(rgb, stored[im.name]) if stored else labeler(rgb)
            seeds = None if key is None else above(key, floor)
            with torch.inference_mode():
                boxes, ms = labeler.timed(self._step, session, tracks, rgb, k, seeds)
            ids = [i for i, (x1, y1, x2, y2) in boxes.items() if x2 > x1 and y2 > y1]
            rows = [[*boxes[i], tracks.live[i].score, tracks.live[i].cls] for i in ids]
            labels = Labels(
                np.array(rows, np.float32).reshape(-1, 6),
                [tracks.live[i].label for i in ids],
                (key.ms if key else {}) | {"tracker": ms},
                key.detected if key else 0,
                np.array(ids, int),
            )
            yield labels, key, len(tracks.live)


def video_tracks(
    labeler: Labeler,
    cfg: Tracker,
    images: list[visdrone_vid.Image],
    raw: list[Labels] | None = None,
    floor: float | None = None,
) -> Iterator[tuple[Labels, Labels | None, int]]:
    """A video-tracker trial over every sequence in ``images``, a new session per sequence.

    ``raw``, the stored detector boxes per image, replaces the detector on key frames.
    """
    tracker = VideoTracker(cfg, labeler.device)
    stored = None if raw is None else {im.name: r for im, r in zip(images, raw, strict=True)}
    for sequence, frames in groupby(images, key=attrgetter("sequence")):
        frames = list(frames)
        peak = 0
        for out in tracker.frames(labeler, frames, stored, floor):
            peak = max(peak, out[2])
            yield out
        logger.info(f"  {sequence}: {len(frames)} frames, peak {peak} live tracks")
    del tracker
    gc.collect()


def box_tracks(
    name: str,
    images: list[visdrone_vid.Image],
    detections: list[Labels],
    floor: float | None = None,
    overrides: dict[str, Any] | None = None,
    workers: int = WORKERS,
) -> list[tuple[Labels, int]]:
    """ultralytics' ``name`` tracker per frame of a cached pass, ``overrides`` over its YAML.

    ``floor`` sets ``FLOOR_KEYS``, which default to YOLO's score scale; the clips run in parallel.
    """
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    args = YAML.load(str(check_yaml(f"{name}.yaml")))
    overrides = overrides or {}
    if unknown := overrides.keys() - (args.keys() - (FLOOR_KEYS if floor is not None else set())):
        raise ValueError(
            f"{name} can't override {sorted(unknown)}: not in its YAML, or the floor's"
        )
    args |= overrides
    if floor is not None:
        args |= dict.fromkeys(FLOOR_KEYS, floor)
    pairs = zip(images, detections, strict=True)
    clips = [list(clip) for _, clip in groupby(pairs, key=lambda pair: pair[0].sequence)]
    context = multiprocessing.get_context("fork")  # the children inherit the imports
    with ProcessPoolExecutor(workers, mp_context=context) as pool:
        tracked = pool.map(partial(track_clip, name, args), clips)
        return [frame for clip in tracked for frame in clip]


def track_clip(
    name: str, args: dict[str, Any], clip: list[tuple[visdrone_vid.Image, Labels]]
) -> list[tuple[Labels, int]]:
    """One clip under a fresh tracker: per frame, the detections it tracked and the live tracks."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.basetrack import BaseTrack
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import IterableSimpleNamespace

    tracker: Any = TRACKER_MAP[name](args=IterableSimpleNamespace(**args))
    BaseTrack.reset_id()  # TrackTrack has no reset_id of its own; the counter is shared
    out = []
    for im, det in clip:
        bgr = np.ascontiguousarray(read_rgb(im.path)[..., ::-1])  # for camera-motion compensation
        start = time.perf_counter()
        rows = tracker.update(Boxes(det.rows, (im.height, im.width)), bgr).reshape(-1, 8)
        ms = (time.perf_counter() - start) * 1000
        idx = rows[:, 7].astype(int)
        labels = Labels(
            det.rows[idx],
            [det.labels[i] for i in idx],
            det.ms | {"tracker": ms},
            det.detected,
            rows[:, 4].astype(int),
        )
        out.append((labels, len(tracker.tracked_stracks) + len(tracker.lost_stracks)))
    return out
