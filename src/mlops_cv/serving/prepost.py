"""The pre/post-processing contract of the exported end2end detection graph.

Single owner of the math that turns an arbitrary image into the exact input tensor the
graph was trained for, and the graph's raw output back into boxes in source-image
coordinates. A service built on this module inherits the accuracy the optimization report
measured — provided the math here agrees with ultralytics' own preprocessing byte for
byte, which is pinned by committed golden fixtures generated with real ultralytics.

The exported graph is the end2end NMS-free head: its output is already final detections
``(1, max_det, 6)`` as ``x1, y1, x2, y2, conf, cls`` in letterbox pixels, so
post-processing is rescale to source coordinates + confidence filter — no NMS decode.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np

# Gray value ultralytics pads letterbox borders with.
PAD_COLOR = 114


class DecodeError(ValueError):
    """The bytes are not a decodable image — the caller's input, not this code's failure."""


def load_image(source: bytes | str | Path | BinaryIO) -> np.ndarray:
    """Decode an image to the HWC uint8 RGB array every function here operates on.

    ``cv2.imdecode`` — the same codec path (and EXIF-orientation behavior) evaluation
    exercised through ``cv2.imread``, so a served upload is decoded exactly the way the
    measured accuracy's inputs were. Raises :class:`DecodeError` on undecodable bytes.
    """
    if isinstance(source, str | Path):
        data = Path(source).read_bytes()
    elif isinstance(source, bytes):
        data = source
    else:
        data = source.read()
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise DecodeError("could not decode the bytes as an image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _letterbox_geometry(
    source_shape: tuple[int, int], letterbox_shape: tuple[int, int]
) -> tuple[float, int, int, int, int]:
    """The one derivation of letterbox placement: ``(gain, new_w, new_h, left, top)``.

    Shared by :func:`letterbox` (to place the resized image) and :func:`scale_boxes`
    (to undo that placement) so the forward and inverse transforms cannot diverge.
    Matches ultralytics' rounding exactly: resized dims rounded per-axis, odd padding
    split by the asymmetric ``round(pad - 0.1)``.
    """
    src_h, src_w = int(source_shape[0]), int(source_shape[1])
    lb_h, lb_w = int(letterbox_shape[0]), int(letterbox_shape[1])
    gain = min(lb_h / src_h, lb_w / src_w)
    new_w, new_h = round(src_w * gain), round(src_h * gain)
    left = round((lb_w - new_w) / 2 - 0.1)
    top = round((lb_h - new_h) / 2 - 0.1)
    return gain, new_w, new_h, left, top


def letterbox(image: np.ndarray, imgsz: int = 640) -> np.ndarray:
    """Aspect-preserving resize onto a centered ``imgsz`` square, gray-padded.

    Byte-for-byte ultralytics ``LetterBox(auto=False, scaleup=True, center=True)`` — the
    inference-time configuration of an exported static graph, including its asymmetric
    ``round(pad ∓ 0.1)`` split of odd padding.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected an HWC uint8 RGB image, got {image.dtype} {image.shape}")
    _, new_w, new_h, left, top = _letterbox_geometry(image.shape[:2], (imgsz, imgsz))
    resized = image
    if (new_h, new_w) != image.shape[:2]:
        # INTER_LINEAR, like ultralytics — pillow/float bilinear are ±1 LSB off it
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.full((imgsz, imgsz, 3), PAD_COLOR, dtype=np.uint8)
    out[top : top + new_h, left : left + new_w] = resized
    return out


def to_input_tensor(letterboxed: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> the graph's ``(1, 3, H, W)`` float32 input, scaled to 0..1."""
    chw = letterboxed.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(chw, dtype=np.float32) / np.float32(255)


def preprocess(image: np.ndarray, imgsz: int = 640) -> np.ndarray:
    """Source HWC uint8 RGB image -> the exact input tensor of the exported graph."""
    return to_input_tensor(letterbox(image, imgsz))


def scale_boxes(
    boxes: np.ndarray, letterbox_shape: tuple[int, int], source_shape: tuple[int, int]
) -> np.ndarray:
    """Rescale xyxy ``boxes`` out of letterbox pixels into source-image pixels.

    Port of ultralytics ``ops.scale_boxes`` (padding subtracted, gain divided out,
    clipped to the source bounds); shapes are ``(height, width)``. Returns a new float32
    array — the input is never mutated.
    """
    gain, _, _, pad_x, pad_y = _letterbox_geometry(source_shape, letterbox_shape)
    src_h, src_w = int(source_shape[0]), int(source_shape[1])
    out = np.array(boxes, dtype=np.float32)
    out[..., [0, 2]] -= pad_x
    out[..., [1, 3]] -= pad_y
    out /= gain
    out[..., [0, 2]] = out[..., [0, 2]].clip(0, src_w)
    out[..., [1, 3]] = out[..., [1, 3]].clip(0, src_h)
    return out


def postprocess(
    output: np.ndarray,
    source_shape: tuple[int, int],
    imgsz: int = 640,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """The graph's raw output -> ``(n, 6)`` detections in source-image pixels.

    ``output`` is the end2end head's ``(1, max_det, 6)`` (or an already-squeezed
    ``(max_det, 6)``) of ``x1, y1, x2, y2, conf, cls`` in letterbox pixels. Detections
    with ``conf >= conf_threshold`` are kept (the boundary is inclusive: a request asking
    for 0.5 receives the 0.5 ones), boxes rescaled and clipped to the source image.
    """
    preds = np.asarray(output, dtype=np.float32)
    if preds.ndim == 3:
        if preds.shape[0] != 1:
            raise ValueError(f"expected a batch of 1, got output shape {preds.shape}")
        preds = preds[0]
    if preds.ndim != 2 or preds.shape[-1] != 6:
        raise ValueError(f"expected (max_det, 6) detections, got shape {preds.shape}")
    kept = preds[preds[:, 4] >= conf_threshold]
    scaled = kept.copy()
    scaled[:, :4] = scale_boxes(kept[:, :4], (imgsz, imgsz), source_shape)
    return scaled
