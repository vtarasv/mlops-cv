"""The synthetic shift is a pure function — pin what it does to the statistics it moves."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from mlops_cv.pipelines.profiling import profile_drift_bytes
from mlops_cv.streaming.shift import Shift


def _frame(seed: int = 0, shade: int = 110, size: tuple[int, int] = (96, 96)) -> bytes:
    """A small textured JPEG: flat colour has no blur signal to collapse."""
    rng = np.random.default_rng(seed)
    pixels = np.clip(rng.normal(shade, 40, (size[1], size[0], 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_defocus_collapses_the_blur_statistic_and_leaves_brightness_alone() -> None:
    """The malignant mode: it destroys detail, which is the reading `blur` measures."""
    frame = _frame()
    before = profile_drift_bytes(frame)
    after = profile_drift_bytes(Shift(mode="defocus", amount=4.0).apply(frame))

    assert after["blur"] < before["blur"] * 0.1, "a 4px defocus must gut the edge variance"
    assert after["brightness"] == pytest.approx(before["brightness"], rel=0.05)


def test_a_larger_defocus_radius_moves_the_statistic_further() -> None:
    """Dialable, not a switch — the demo turns the knob until the panel crosses."""
    frame = _frame()
    radii = (0.5, 1.0, 2.0)
    blur = [
        profile_drift_bytes(Shift(mode="defocus", amount=r).apply(frame))["blur"] for r in radii
    ]

    assert blur == sorted(blur, reverse=True), f"blur must fall monotonically with radius: {blur}"


def test_brightness_scales_the_brightness_statistic_by_about_the_amount() -> None:
    """The benign mode: a multiplier on the pixels is a multiplier on the grayscale mean."""
    frame = _frame()
    before = profile_drift_bytes(frame)
    after = profile_drift_bytes(Shift(mode="brightness", amount=0.55).apply(frame))

    assert after["brightness"] == pytest.approx(0.55 * before["brightness"], rel=0.1)


def test_the_shifted_frame_is_still_a_decodable_jpeg_of_the_same_size() -> None:
    """It goes on the wire as a frame like any other — the consumers must not notice."""
    frame = _frame(size=(64, 48))
    shifted = Shift(mode="defocus", amount=3.0).apply(frame)

    with Image.open(io.BytesIO(shifted)) as im:
        im.load()
        assert im.format == "JPEG"
        assert im.size == (64, 48)


def test_an_unusable_shift_is_refused_rather_than_published() -> None:
    with pytest.raises(ValueError, match="brightness"):  # the known modes are named in the message
        Shift(mode="sharpen", amount=1.0)
    with pytest.raises(ValueError, match="positive"):
        Shift(mode="defocus", amount=0.0)


def test_the_shift_describes_itself_as_synthetic() -> None:
    """Wherever a shift is announced it must say it is simulated — the log line uses this."""
    described = Shift(mode="defocus", amount=4.0).describe()

    assert "synthetic" in described.lower()
    assert "4" in described
