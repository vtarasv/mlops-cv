"""Regenerate ``drift_golden.json`` — the measured separation between normal and shifted footage.

The committed fixture is *measurement*, not expectation: a real training-scene cloud (the
published drift baseline of the profiled subset) and real window means, profiled off the demo
store with the very function that produced the baseline. The expected scores and verdicts live in
``tests/unit/test_drift_golden.py`` as literals, so a refactor that quietly stops detecting is a
red test rather than a regenerated fixture.

Run locally, against a profiled subset with a demo store, only when the subset, the sampling
stride or the drift statistics change — the same remeasure obligation the numbers themselves
carry::

    uv run python tests/unit/fixtures/generate_drift_golden.py

Shift modes are the demo's simulated shifts: a defocus (measurably malignant — it destroys the
champion's detections) and two photometric multipliers (measurably benign — the model absorbs
them). Each shifted frame is re-encoded to JPEG before profiling, because that is what a shifted
frame on the bus is.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageFilter

from mlops_cv.config import get_settings
from mlops_cv.monitoring.drift import window_means
from mlops_cv.pipelines import profiling
from mlops_cv.streaming.producer import discover_sequences

FIXTURE = Path(__file__).parent / "drift_golden.json"

SAMPLE_EVERY = 5  # the monitor's sampling stride
MAX_WINDOW = 100  # sampled frames per window
JPEG_QUALITY = 95


def _multiply(factor: float) -> Callable[[Image.Image], Image.Image]:
    return lambda im: Image.eval(im, lambda value: min(255, int(value * factor)))


SHIFTS: dict[str, Callable[[Image.Image], Image.Image] | None] = {
    "normal": None,
    "defocus-r4": lambda im: im.filter(ImageFilter.GaussianBlur(4)),
    "dark-x0.55": _multiply(0.55),
    "bright-x1.6": _multiply(1.6),
}


def shifted_bytes(data: bytes, shift: Callable[[Image.Image], Image.Image] | None) -> bytes:
    """Apply one simulated shift and re-encode, the way a shifted frame reaches the bus."""
    if shift is None:
        return data
    with Image.open(io.BytesIO(data)) as im:
        buf = io.BytesIO()
        shift(im.convert("RGB")).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def window(frames: list[Path], shift: Callable[[Image.Image], Image.Image] | None) -> dict:
    """Profile one clip's sampled window, exactly as the monitor samples a live stream."""
    sampled = frames[::SAMPLE_EVERY][:MAX_WINDOW]
    readings = [
        profiling.profile_image_bytes(shifted_bytes(frame.read_bytes(), shift)) for frame in sampled
    ]
    return {"n_frames": len(readings), "means": window_means(readings)}


def main() -> int:
    settings = get_settings()
    subset = settings.data.subset_dir
    scenes = profiling.read_baseline(subset / profiling.PROFILE_DIRNAME)
    # The demo store as the producer replays it — same clips, same frame order.
    clips = discover_sequences(subset, None)

    windows = []
    for clip, frames in clips.items():
        for mode, shift in SHIFTS.items():
            measured = window(frames, shift)
            windows.append({"clip": clip, "mode": mode, **measured})
            print(f"{clip} {mode}: {measured['means']}")

    FIXTURE.write_text(
        json.dumps(
            {
                "source": {
                    "subset": subset.name,
                    "sample_every": SAMPLE_EVERY,
                    "max_window": MAX_WINDOW,
                    "jpeg_quality": JPEG_QUALITY,
                },
                "baseline": [scene.model_dump() for scene in scenes],
                "windows": windows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {FIXTURE} ({len(scenes)} training scenes, {len(windows)} windows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
