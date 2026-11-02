"""Track scores through TrackEval's MOTChallenge adapter, once per merged class, glue only.

``MotChallenge2DBox`` scores a single class and never filters tracker rows by class, so rows are
split by class before they are written. Both sides arrive already through VisDrone's
ignored-region rule. The classes combine det-averaged: a class absent from a sequence would score
0 in a plain class average.
"""

from __future__ import annotations

import contextlib
import io
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from mlops_cv.data.convert_visdrone_vid import YOLO_NAMES
from mlops_cv.tuning.autolabel_visdrone_vid import visdrone_vid

TRACK = "track"
# TrackEval metric -> the fields reported; HOTA's are means over its localisation thresholds.
FIELDS = {"HOTA": ("HOTA", "DetA", "AssA"), "Identity": ("IDF1",), "CLEAR": ("IDSW",)}


def mot_files(
    images: list[visdrone_vid.Image], tracked: list[np.ndarray]
) -> tuple[dict[tuple[int, str, str], list[str]], dict[str, int]]:
    """``{(class, gt|trk, sequence): MOT lines}`` and ``{sequence: frames}``.

    ``tracked`` holds each frame's ``(n, 7)`` x1 y1 x2 y2 conf class track-id rows. Frames are
    numbered 1.. within the sequence's frames at hand.
    """
    lines: dict[tuple[int, str, str], list[str]] = defaultdict(list)
    seq_info: dict[str, int] = {}
    pairs = zip(images, tracked, strict=True)
    for seq, frames in groupby(pairs, key=lambda pair: pair[0].sequence):
        frames = list(frames)
        seq_info[seq] = len(frames)
        for t, (im, rows) in enumerate(frames, start=1):
            gt = zip(visdrone_vid.merged(im.categories), im.track_ids, im.boxes, strict=True)
            for c, i, (x, y, w, h) in gt:
                lines[c, "gt", seq].append(f"{t},{i},{x},{y},{w},{h},1,1,1")
            for x1, y1, x2, y2, conf, c, i in rows:
                trk = f"{t},{int(i)},{x1},{y1},{x2 - x1},{y2 - y1},{conf},-1,-1,-1"
                lines[int(c), "trk", seq].append(trk)
    return lines, seq_info


def class_results(root: Path, seq_info: dict[str, int], metrics: list[Any]) -> dict[str, Any]:
    """One class's MOT files under ``root`` -> TrackEval's combined-sequence results per metric."""
    import trackeval

    dataset = trackeval.datasets.MotChallenge2DBox(
        {
            "GT_FOLDER": str(root / "gt"),
            "TRACKERS_FOLDER": str(root / "trk"),
            "OUTPUT_FOLDER": str(root / "out"),
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": seq_info,
            "BENCHMARK": "VisDrone",
            "DO_PREPROC": False,
            "PRINT_CONFIG": False,
        }
    )
    quiet = {"PRINT_RESULTS": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False}
    off = {"OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False}
    evaluator = trackeval.Evaluator(quiet | off | {"LOG_ON_ERROR": None})
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = evaluator.evaluate([dataset], metrics)
    return res["MotChallenge2DBox"]["t"]["COMBINED_SEQ"]["pedestrian"]


def summary(res: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}/{field}": float(np.mean(res[metric][field]))
        for metric, fields in FIELDS.items()
        for field in fields
    }


def readings(images: list[visdrone_vid.Image], tracked: list[np.ndarray]) -> dict[str, float]:
    """HOTA (DetA, AssA), IDF1 and ID switches, det-averaged and per merged class."""
    import trackeval

    lines, seq_info = mot_files(images, tracked)
    quiet = {"PRINT_CONFIG": False}
    metrics = {
        "HOTA": trackeval.metrics.HOTA(),
        "CLEAR": trackeval.metrics.CLEAR(quiet),
        "Identity": trackeval.metrics.Identity(quiet),
    }
    per_class: dict[str, dict[str, Any]] = {}
    with TemporaryDirectory() as tmp:
        for c, name in YOLO_NAMES.items():
            root = Path(tmp) / name
            for seq in seq_info:
                for path, kind in ((f"gt/{seq}/gt/gt.txt", "gt"), (f"trk/t/data/{seq}.txt", "trk")):
                    (root / path).parent.mkdir(parents=True, exist_ok=True)
                    (root / path).write_text("\n".join(lines.get((c, kind, seq), [])))
            per_class[name] = class_results(root, seq_info, list(metrics.values()))
    combined = {
        name: metric.combine_classes_det_averaged({c: res[name] for c, res in per_class.items()})
        for name, metric in metrics.items()
    }
    out = summary(combined, TRACK)
    for name, res in per_class.items():
        out |= summary(res, f"{TRACK}/{name}")
    return out
