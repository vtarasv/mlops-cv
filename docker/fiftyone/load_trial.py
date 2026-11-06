"""Load a labeling trial's auto-labels into the FiftyOne App.

Runs inside the App container (docs/tuning/autolabel-visdrone-vid.md), so the client always
matches the server. Both sides are VisDrone-VID annotation files: the raw split's ground truth,
where a sequence has one, and the trial's, one per sequence.
One dataset per raw split grows by the frames each trial labels.
"""

from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# mlops_cv.data's module, copied beside this script: the image has no mlops_cv.
from convert_visdrone_vid import CLASS_MERGE, RAW_NAMES, YOLO_NAMES

GROUND_TRUTH, IGNORED = "ground_truth", "ignored"
IGNORED_CATEGORY = 0


def dataset_name(raw_split: str) -> str:
    return f"visdrone-vid-{raw_split.removeprefix('VisDrone2019-VID-')}"


def field_name(run_name: str) -> str:
    """The run name as a FiftyOne field name (no dots or dashes)."""
    return re.sub(r"\W", "_", run_name)


def read_annotations(text: str) -> dict[int, list[list[float]]]:
    """A VisDrone-VID annotation file's text -> ``{frame: rows}``, each row's 10 columns."""
    frames: dict[int, list[list[float]]] = defaultdict(list)
    for line in text.splitlines():
        if line.strip():
            row = [float(v) for v in line.split(",")[:10]]
            frames[int(row[0])].append(row)
    return dict(frames)


def ground_truth(raw_split: Path, sequence: str) -> dict[int, list[list[float]]] | None:
    """The sequence's ground truth by frame, or ``None`` for footage without an annotation file."""
    path = raw_split / "annotations" / f"{sequence}.txt"
    return read_annotations(path.read_text()) if path.exists() else None


def boxes(rows: list[list[float]], size: tuple[int, int], scored: bool) -> list[dict[str, Any]]:
    """``fo.Detection`` kwargs for the rows of a merged class; ``scored`` sets the confidence."""
    w, h = size
    return [
        {
            "label": YOLO_NAMES[CLASS_MERGE[int(r[7])]],
            "bounding_box": [r[2] / w, r[3] / h, r[4] / w, r[5] / h],
            "confidence": r[6] if scored else None,
            "raw_category": RAW_NAMES[int(r[7])],
            "track_id": int(r[1]),
        }
        for r in rows
        if int(r[7]) in CLASS_MERGE
    ]


def ignored(rows: list[list[float]], size: tuple[int, int]) -> list[dict[str, Any]]:
    w, h = size
    return [
        {"label": IGNORED, "bounding_box": [r[2] / w, r[3] / h, r[4] / w, r[5] / h]}
        for r in rows
        if int(r[7]) == IGNORED_CATEGORY
    ]


def main(raw_split: Path, trial: Path) -> None:
    import fiftyone as fo
    from PIL import Image

    def detections(kwargs: list[dict[str, Any]]) -> Any:
        return fo.Detections(detections=[fo.Detection(**k) for k in kwargs])

    start = time.perf_counter()
    name = dataset_name(raw_split.name)
    ds = fo.load_dataset(name) if fo.dataset_exists(name) else fo.Dataset(name, persistent=True)
    have = set(ds.values("filepath"))
    field = field_name(trial.parent.name)  # runs/autolabel/<run name>/annotations
    # ponytail: a frame a trial labeled with no box has no line, so it isn't loaded.
    values, new, annotated = {}, [], []
    for path in sorted(trial.glob("*.txt")):
        seq, frames_dir = path.stem, raw_split / "sequences" / path.stem
        with Image.open(min(frames_dir.glob("*.jpg"))) as im:
            size = im.size
        truth = ground_truth(raw_split, seq)
        for frame, rows in read_annotations(path.read_text()).items():
            filepath = str(frames_dir / f"{frame:07d}.jpg")
            if truth is not None:
                annotated.append(filepath)
            if filepath not in have:
                gt = (truth or {}).get(frame, [])
                new.append(
                    fo.Sample(
                        filepath=filepath,
                        metadata=fo.ImageMetadata(width=size[0], height=size[1]),
                        sequence=seq,
                        frame_number=frame,
                        **{
                            GROUND_TRUTH: detections(boxes(gt, size, scored=False)),
                            IGNORED: detections(ignored(gt, size)),
                        },
                    )
                )
            values[filepath] = detections(boxes(rows, size, scored=True))
    ds.add_samples(new)
    eval_key = f"eval_{field}"
    if ds.has_evaluation(eval_key):
        ds.delete_evaluation(eval_key)
    if ds.has_sample_field(field):
        ds.delete_sample_field(field)
    ds.set_values(field, values, key_field="filepath")
    if annotated:  # footage without ground truth isn't evaluated: every box would be FP
        # Browsing only: these flags count every confidence and match unlike the validator.
        view = ds.select_by("filepath", annotated)
        view.evaluate_detections(field, GROUND_TRUTH, eval_key=eval_key, iou=0.5)
    print(f"{name}: {field} on {len(values)} frames in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
