"""Regenerate ``prepost_golden.npz`` from real ultralytics preprocessing.

The committed fixture is the serving pre/post contract: CI pins
``mlops_cv.serving.prepost`` against it without installing ultralytics (the fixture is
the contract, like the emit<->parse pins). Run this locally, where the ``gpu`` group is
installed, only when ultralytics itself changes the math being mirrored:

    uv run python tests/unit/fixtures/generate_prepost_golden.py

Every golden is produced by ultralytics code (``LetterBox``, the predictor's
normalization steps, ``ops.scale_boxes``) — never by the module under test — and the
script refuses to write a fixture the module disagrees with, so a regeneration can
tighten the contract but never quietly loosen it to whatever the module currently does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).parent / "prepost_golden.npz"

# (name, source height, source width, letterbox target, noise): downscale, upscale
# (smaller-than-target), cv2's rerouted exact-2x downscale, extreme aspect, and one
# VisDrone-like aspect at the real 640 target. The small cases carry noise to exercise
# the fixed-point rounding; the 640 case stays smooth so the committed npz stays small.
LETTERBOX_CASES = [
    ("down", 76, 120, 64, True),
    ("up", 28, 40, 64, True),
    ("half", 128, 128, 64, True),
    ("narrow", 100, 30, 96, True),
    ("visdrone", 359, 639, 640, False),
]

# Letterbox-space xyxy boxes for the scale_boxes goldens: interior, pad-overlapping
# (clips negative), full-frame, and sub-pixel coordinates.
SCALE_BOXES_IN = [
    [100.0, 200.0, 400.0, 500.0],
    [0.0, 100.0, 640.0, 550.0],
    [2.5, 141.25, 637.5, 498.75],
    [300.7, 139.2, 301.1, 645.0],
]
SCALE_BOXES_LETTERBOX = (640, 640)
# Two source shapes: one with exact gains, and one whose non-limiting dimension rounds
# (654 * 0.64 = 418.56 -> 419) so the fixture pins ultralytics' inner round(src * gain) —
# with the round omitted the derived padding comes out one pixel off.
SCALE_BOXES_SOURCES = {"scale_boxes": (756, 1344), "scale_boxes_rounding": (654, 1000)}


def _source_image(rng: np.random.Generator, height: int, width: int, noise: bool) -> np.ndarray:
    """Diagonal gradient, optionally plus noise (which exercises the rounding paths)."""
    yy, xx = np.mgrid[0:height, 0:width]
    base = ((yy * 255 / max(height - 1, 1) + xx * 191 / max(width - 1, 1)) / 2).astype(np.uint8)
    img = np.repeat(base[..., None], 3, axis=2)
    if noise:
        # uint8 wraparound is fine — any bytes are valid pixels
        img = img + rng.integers(0, 32, size=(height, width, 3), dtype=np.uint8)
    return img


def main() -> None:
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops

    from mlops_cv.serving import prepost

    rng = np.random.default_rng(20260808)
    payload: dict[str, np.ndarray] = {}

    for name, height, width, imgsz, noise in LETTERBOX_CASES:
        src = _source_image(rng, height, width, noise)
        # ultralytics letterboxes the BGR frame, then the predictor flips it to RGB;
        # the transform is channel-independent, so feeding RGB directly is the same
        # contract our RGB-in module implements.
        golden = LetterBox((imgsz, imgsz), auto=False, scaleup=True, center=True)(image=src)
        assert np.array_equal(prepost.letterbox(src, imgsz), golden), f"module diverges: {name}"  # type: ignore
        payload[f"{name}_src"] = src
        payload[f"{name}_letterboxed"] = golden  # type: ignore
        payload[f"{name}_imgsz"] = np.array(imgsz)

    # The predictor's normalization (BHWC->BCHW, float32, /255) on the smallest case.
    letterboxed = payload["down_letterboxed"]
    tensor = torch.from_numpy(np.ascontiguousarray(letterboxed[None].transpose(0, 3, 1, 2))).float()
    tensor /= 255
    golden_tensor = tensor.numpy()
    assert np.array_equal(prepost.to_input_tensor(letterboxed), golden_tensor)
    payload["down_tensor"] = golden_tensor

    boxes_in = np.array(SCALE_BOXES_IN, dtype=np.float32)
    payload["scale_boxes_in"] = boxes_in
    payload["scale_boxes_letterbox_shape"] = np.array(SCALE_BOXES_LETTERBOX)
    for name, source_shape in SCALE_BOXES_SOURCES.items():
        golden_boxes = ops.scale_boxes(
            SCALE_BOXES_LETTERBOX, torch.from_numpy(boxes_in.copy()), source_shape
        ).numpy()  # type: ignore
        ours = prepost.scale_boxes(boxes_in, SCALE_BOXES_LETTERBOX, source_shape)
        assert np.array_equal(ours, golden_boxes), f"{name} diverges from ultralytics"
        payload[f"{name}_out"] = golden_boxes
        payload[f"{name}_source_shape"] = np.array(source_shape)

    np.savez_compressed(FIXTURE, **payload)  # type: ignore
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
