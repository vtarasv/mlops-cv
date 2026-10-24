"""COCO small-object AP (mAP50-95_small) over the validator's own COCO-JSON predictions: glue only.

The AP math is ``faster-coco-eval``'s, the library ultralytics itself calls for ``mAP_small``.
This module only bridges the subset's YOLO labels into a COCO ground truth with the ids
ultralytics' ``pred_to_json`` writes (class ``c`` -> category ``c + 1``, image -> file stem).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# COCO keeps 100 boxes per image; val keeps 300, so mAP50-95_small sees the same boxes as mAP50-95.
MAX_DETS = 300


def coco_ground_truth(
    labels: Iterable[Mapping[str, Any]], names: Mapping[int, str]
) -> tuple[dict[str, Any], dict[str, int]]:
    """Ultralytics dataset labels -> ``(COCO ground-truth dict, {file stem: image id})``.

    Each label carries ``im_file``, ``shape`` (h, w source pixels), ``cls`` (n, 1) and
    ``bboxes`` (n, 4) normalized xywh. Box areas are in source pixels, where COCO's 32² small
    bucket is defined.
    """
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ids: dict[str, int] = {}
    for label in labels:
        stem = Path(label["im_file"]).stem
        if stem in ids:
            raise ValueError(f"file stem {stem} repeats; COCO image ids must be unique")
        ids[stem] = image_id = len(ids) + 1
        h, w = label["shape"]
        images.append({"id": image_id, "width": w, "height": h, "file_name": stem})
        for c, (x, y, bw, bh) in zip(label["cls"].reshape(-1), label["bboxes"], strict=True):
            bw, bh = float(bw) * w, float(bh) * h
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": int(c) + 1,
                    "bbox": [float(x) * w - bw / 2, float(y) * h - bh / 2, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
    categories = [{"id": i + 1, "name": n} for i, n in sorted(names.items())]
    return {"images": images, "annotations": annotations, "categories": categories}, ids


def coco_stats(
    predictions: Sequence[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
    names: Mapping[int, str],
) -> dict[str, float]:
    """``faster-coco-eval``'s bbox stats (``AP_all``, ``AP_small``, ...) at ``maxDets`` 300.

    ``predictions`` is ultralytics' ``predictions.json`` content. Its ``image_id`` is the stem,
    turned into an int when numeric, so predictions are re-keyed by ``file_name`` instead.
    """
    from faster_coco_eval import COCO, COCOeval_faster

    gt, ids = coco_ground_truth(labels, names)
    if not predictions:
        return {"AP_all": 0.0, "AP_small": 0.0}
    dets = [{**p, "image_id": ids[Path(p["file_name"]).stem]} for p in predictions]
    coco_gt = COCO(gt)
    evaluator = COCOeval_faster(coco_gt, coco_gt.loadRes(dets), iouType="bbox")
    evaluator.params.maxDets = [1, 10, MAX_DETS]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {k: float(v) for k, v in evaluator.stats_as_dict.items()}


def ap_small(
    predictions: Sequence[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
    names: Mapping[int, str],
) -> float:
    """COCO AP over boxes under 32² source pixels, IoU 0.50:0.95, all classes."""
    return coco_stats(predictions, labels, names)["AP_small"]


def load_predictions(save_dir: str | Path) -> list[dict[str, Any]]:
    """A ``model.val(save_json=True)`` run's ``predictions.json``; ``[]`` when it wrote none."""
    path = Path(save_dir) / "predictions.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
