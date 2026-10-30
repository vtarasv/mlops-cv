"""Reported only: box-prompted segmenters prompted with the ground-truth boxes, never selecting.

Per segmenter, on a scoring set: how well each mask's tight box fits the GT box that prompted it,
and, with ``--aerovis``, the mask IoU against AeroVIS's reviewed masks on the frames it holds byte
for byte (VID train frames, VID's own track ids). AeroVIS stays local: its folder is an argument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mlops_cv.autolabel import visdrone_vid
from mlops_cv.autolabel.labeler import box_prompted, mask_areas, read_rgb, torch_device
from mlops_cv.autolabel.run import AUTOLABEL_EXPERIMENT, SET_TAG
from mlops_cv.startup import exits_on_startup_error

logger = logging.getLogger(__name__)

SEGMENTERS = ("facebook/sam2.1-hiera-small", "facebook/sam2.1-hiera-large", "facebook/sam3")
FIT_IOUS = (0.5, 0.75)


def pair_iou(a: list[Any] | np.ndarray, b: list[Any] | np.ndarray) -> np.ndarray:
    """pycocotools' IoU of each ``a[i]`` with ``b[i]``: RLEs or xywh boxes."""
    from pycocotools import mask

    mask_utils: Any = mask
    if not len(a):
        return np.zeros(0)
    return np.diag(np.asarray(mask_utils.iou(a, b, [0] * len(b))))


def fit_readings(ious: np.ndarray, empty: np.ndarray) -> dict[str, float]:
    """Mean box IoU and F1 at each ``FIT_IOUS``; an empty mask is a miss, not a prediction."""
    out = {"fit/box_iou_mean": float(ious.mean()), "fit/empty_masks": float(empty.sum())}
    for t in FIT_IOUS:
        hits = (ious >= t).sum()
        out[f"fit/F1_iou{round(t * 100)}"] = float(2 * hits / ((~empty).sum() + len(ious)))
    return out


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def aerovis_masks(root: Path, images: list[visdrone_vid.Image]) -> dict[str, dict[int, Any]]:
    """``{frame name: {VID track id: AeroVIS RLE}}`` for the frames AeroVIS holds byte for byte."""
    by_size: dict[int, list[visdrone_vid.Image]] = defaultdict(list)
    for im in images:
        by_size[im.path.stat().st_size].append(im)
    data = json.loads((root / "aero_vis.json").read_text(encoding="utf-8"))
    frames: dict[tuple[int, int], str] = {}  # (video id, frame index) -> frame name
    for video in data["videos"]:
        for k, name in enumerate(video["file_names"]):
            path = root / "sequences" / name
            if candidates := by_size.get(path.stat().st_size):
                digest = md5(path)
                if hit := next((im for im in candidates if md5(im.path) == digest), None):
                    frames[video["id"], k] = hit.name
    masks: dict[str, dict[int, Any]] = defaultdict(dict)
    for ann in data["annotations"]:
        if ann.get("iscrowd"):
            continue
        for k, rle in enumerate(ann["segmentations"]):
            if rle and (name := frames.get((ann["video_id"], k))):
                masks[name][ann["track_id"]] = rle
    return dict(masks)


def xyxy(boxes: np.ndarray) -> np.ndarray:
    return np.column_stack([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]])


def check(
    repo: str,
    images: list[visdrone_vid.Image],
    reference: dict[str, dict[int, Any]],
    device: str,
) -> dict[str, float]:
    """One segmenter prompted with each frame's GT boxes: its box fit and its mask IoU."""
    from pycocotools import mask

    mask_utils: Any = mask
    segment = box_prompted(repo, torch_device(device))
    ious, empty, mask_ious = [], [], []
    for i, im in enumerate(images, start=1):
        rles = segment(read_rgb(im.path), xyxy(im.boxes))
        if rles:
            ious.append(pair_iou(mask_utils.toBbox(rles), im.boxes))
            empty.append(mask_areas(rles) == 0)
        ref = reference.get(im.name, {})
        pairs = [(r, ref[t]) for r, t in zip(rles, im.track_ids.tolist(), strict=True) if t in ref]
        mask_ious.append(pair_iou([p for p, _ in pairs], [a for _, a in pairs]))
        if i % 50 == 0:
            logger.info(f"  {i}/{len(images)} frames")
    out = fit_readings(np.concatenate(ious), np.concatenate(empty))
    if reference:
        matched = np.concatenate(mask_ious)
        out |= {
            "aerovis/mask_iou_mean": float(matched.mean()),
            "aerovis/masks": float(len(matched)),
            "aerovis/frames": float(sum(im.name in reference for im in images)),
        }
    return out


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prompt each segmenter with the ground-truth boxes.")
    p.add_argument("--set", choices=visdrone_vid.SETS, default="train-sample", dest="scoring_set")
    p.add_argument("--aerovis", type=Path, help="the local AeroVIS folder (aero_vis.json)")
    args = p.parse_args(argv)

    os.environ["MLFLOW__EXPERIMENT"] = AUTOLABEL_EXPERIMENT
    from mlops_cv.config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    data = settings.data
    images = visdrone_vid.prepare(data.raw_dir, data.autolabel_dir, args.scoring_set)
    reference = aerovis_masks(args.aerovis, images) if args.aerovis else {}
    logger.info(f"AeroVIS masks on {len(reference)} of {len(images)} frames")

    from mlops_cv.tracking import client

    mlflow = client.connect(settings)
    for repo in SEGMENTERS:
        logger.info(f"{repo}: prompting with the GT boxes of {len(images)} frames")
        metrics = check(repo, images, reference, settings.training.device)
        with mlflow.start_run(run_name=f"gt-box-check-{repo.split('/')[-1]}-{args.scoring_set}"):
            mlflow.set_tags({"autolabel.stage": "gt-box-check", SET_TAG: args.scoring_set})
            mlflow.log_params({"segmenter": repo, "frames": len(images)})
            mlflow.log_metrics(metrics)
        logger.info(f"{repo}: {json.dumps(metrics, indent=1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
