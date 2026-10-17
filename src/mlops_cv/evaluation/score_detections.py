"""Score stored detections with ultralytics' own validator metric code: glue only, no AP math."""

from __future__ import annotations

import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ultralytics.utils.metrics import DetMetrics


def score(detections: Mapping[str, np.ndarray], data_yaml: Path, split: str) -> DetMetrics:
    """``{frame file name: (n, 6) x1 y1 x2 y2 conf cls, source pixels}`` -> the validator's metrics.

    Each frame goes through ``DetectionValidator.update_metrics`` with a batch whose image shape
    is the source shape, so ``_prepare_batch`` scales the normalized labels into source pixels,
    where the detections live.
    """
    import torch
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.yolo.detect import DetectionValidator

    with tempfile.TemporaryDirectory() as save_dir:
        validator = DetectionValidator(
            save_dir=Path(save_dir), args={"data": str(data_yaml), "split": split, "plots": False}
        )
        validator.data = check_det_dataset(str(data_yaml), autodownload=False)
        validator.stride = 32
        labels = validator.build_dataset(validator.data[split]).labels  # type: ignore[attr-defined]
        names = Counter(Path(lb["im_file"]).name for lb in labels)
        if dupes := sorted(n for n, c in names.items() if c > 1):
            raise ValueError(
                f"{len(dupes)} {split} file names repeat across directories, and detections are "
                f"keyed by bare file name: {', '.join(dupes[:5])}"
            )
        if missing := sorted(names.keys() - set(detections)):
            raise ValueError(
                f"{len(missing)} {split} frames have no detections (an empty prediction is an "
                f"empty array, not an absent frame): {', '.join(missing[:5])}"
            )
        validator.init_metrics(SimpleNamespace(names=validator.data["names"]))  # type: ignore[arg-type]
        for label in labels:
            h, w = label["shape"]
            n = len(label["cls"])
            batch = {
                "batch_idx": torch.zeros(n),
                "cls": torch.from_numpy(label["cls"]).float(),
                "bboxes": torch.from_numpy(label["bboxes"]).float(),
                "ori_shape": [(h, w)],
                "ratio_pad": [None],
                "img": SimpleNamespace(shape=(1, 3, h, w)),  # source pixels: no letterbox to undo
                "im_file": [label["im_file"]],
            }
            rows = torch.as_tensor(detections[Path(label["im_file"]).name]).float().reshape(-1, 6)
            pred = {"bboxes": rows[:, :4], "conf": rows[:, 4], "cls": rows[:, 5]}
            validator.update_metrics([pred], batch)
        validator.get_stats()
    return validator.metrics
