"""Torch-free NCNN inference: ultralytics' NCNN predict in ncnn + numpy, on `serving.prepost`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from mlops_cv.serving import prepost

MAX_NMS = 30000  # ultralytics' candidate cap before NMS
MAX_WH = 7680  # ultralytics' per-class box offset, so one NMS pass never crosses classes


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy NMS keeping IoU <= ``iou``, as ``TorchNMS.nms`` / torchvision do."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        w = (np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest])).clip(0)
        h = (np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest])).clip(0)
        inter = w * h
        with np.errstate(invalid="ignore"):  # 0/0 between two zero-area boxes
            overlap = inter / (areas[i] + areas[rest] - inter)
        order = rest[~(overlap > iou)]  # NaN IoU keeps the box, as torchvision does
    return np.array(keep, dtype=np.int64)


def decode(
    output: np.ndarray,
    conf: float,
    iou: float,
    *,
    max_det: int = 300,
    multi_label: bool = False,
) -> np.ndarray:
    """NCNN ``out0`` ``(4 + nc, anchors)`` -> ``(n, 6)`` xyxy/conf/cls in letterbox pixels.

    Port of ultralytics ``non_max_suppression`` (``utils/nms.py:72-157``).
    """
    if output.ndim != 2 or output.shape[0] < 5:
        raise ValueError(f"expected (4 + nc, anchors) output, got shape {output.shape}")
    x = output.T
    xy, wh = x[:, :2], x[:, 2:4]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1)
    scores = x[:, 4:]
    if multi_label:
        rows, cls = np.nonzero(scores > conf)
        boxes, best = boxes[rows], scores[rows, cls]
    else:
        cls = scores.argmax(1)
        best = scores[np.arange(len(cls)), cls]
        kept = best > conf
        boxes, best, cls = boxes[kept], best[kept], cls[kept]
    if len(best) > MAX_NMS:
        top = np.argsort(-best, kind="stable")[:MAX_NMS]
        boxes, best, cls = boxes[top], best[top], cls[top]
    keep = _nms(boxes + cls[:, None].astype(np.float32) * MAX_WH, best, iou)[:max_det]
    return np.column_stack([boxes[keep], best[keep], cls[keep]]).astype(np.float32)


class NcnnDetector:
    """One NCNN export dir (``*.param`` + ``*.bin`` + ``metadata.yaml``) at ``imgsz``."""

    def __init__(self, model_dir: Path, imgsz: int) -> None:
        import ncnn

        self.imgsz = imgsz
        metadata = yaml.safe_load((model_dir / "metadata.yaml").read_text())
        self.names: dict[int, str] = metadata["names"]
        self.net = ncnn.Net()  # type: ignore[attr-defined]
        # As ultralytics' backend: every other Option stays default, so fp16 numerics match.
        self.net.opt.use_vulkan_compute = False
        param = next(model_dir.glob("*.param"))
        self.net.load_param(str(param))
        self.net.load_model(str(param.with_suffix(".bin")))
        self._input = self.net.input_names()[0]
        (self._output,) = self.net.output_names()
        self._mat = ncnn.Mat  # type: ignore[attr-defined]

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        """``(1, 3, imgsz, imgsz)`` input tensor -> the graph's raw ``out0``."""
        with self.net.create_extractor() as ex:
            ex.input(self._input, self._mat(tensor[0]))
            return np.array(ex.extract(self._output)[1])

    def predict(
        self,
        image_rgb: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        multi_label: bool = False,
    ) -> np.ndarray:
        """HWC uint8 RGB image -> ``(n, 6)`` xyxy/conf/cls in source pixels."""
        raw = self.forward(prepost.preprocess(image_rgb, self.imgsz))
        dets = decode(raw, conf, iou, max_det=max_det, multi_label=multi_label)
        dets[:, :4] = prepost.scale_boxes(
            dets[:, :4], (self.imgsz, self.imgsz), image_rgb.shape[:2]
        )
        return dets
