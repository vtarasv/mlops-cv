"""Per-frame CV quality metrics for dataset profiling.

Flag thresholds are **informational**: they mark frames in the quality report for a human (or a
future drift monitor) to look at — only corrupt/unreadable images fail the profiling run.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

from mlops_cv.data.convert_visdrone_vid import YoloBox
from mlops_cv.data.subset import MANIFEST_FILENAME
from mlops_cv.pipelines import provenance

PROFILE_DIRNAME = "profile"
PROFILE_JSON = "profile.json"
QUALITY_REPORT = "quality_report.csv"
STAMP_FILENAME = ".profile-stamp.json"

DHASH_SIZE = 8  # 8x8 difference hash -> 64-bit hex string

# Informational flag thresholds (8-bit grayscale units; quality-report rows, never failures).
FLAG_DARK_MEAN = 20.0
FLAG_BRIGHT_MEAN = 235.0
FLAG_LOW_CONTRAST_STD = 10.0
FLAG_BLUR_VAR = 50.0  # variance of the edge image below this -> soft/blurry frame

# The parameters a profile is stamped with — changing any of them invalidates existing profiles.
PROFILE_PARAMS: dict = {
    "dhash_size": DHASH_SIZE,
    "dark_mean": FLAG_DARK_MEAN,
    "bright_mean": FLAG_BRIGHT_MEAN,
    "low_contrast_std": FLAG_LOW_CONTRAST_STD,
    "blur_var": FLAG_BLUR_VAR,
}


def brightness_contrast(im: Image.Image) -> tuple[float, float]:
    """Grayscale mean (brightness) and stddev (contrast) of an image."""
    stat = ImageStat.Stat(im.convert("L"))
    return stat.mean[0], stat.stddev[0]


def blur_score(im: Image.Image) -> float:
    """Variance of the edge-filtered grayscale image — lower means softer/blurrier."""
    return ImageStat.Stat(im.convert("L").filter(ImageFilter.FIND_EDGES)).var[0]


def dhash(im: Image.Image, size: int = DHASH_SIZE) -> str:
    """Difference hash: near-duplicate frames get identical hex strings."""
    import imagehash  # lazy: keeps the module importable where the beam group isn't installed

    return str(imagehash.dhash(im, hash_size=size))


def profile_image_bytes(data: bytes) -> dict:
    """Decode one image and compute its per-frame metrics; raises on corrupt/unreadable data."""
    Image.open(io.BytesIO(data)).verify()  # structural check; verify() invalidates the handle
    with Image.open(io.BytesIO(data)) as im:
        im.load()  # force a full decode so truncated data fails here, not lazily later
        width, height = im.size
        brightness, contrast = brightness_contrast(im)
        return {
            "width": width,
            "height": height,
            "brightness": brightness,
            "contrast": contrast,
            "blur": blur_score(im),
            "dhash": dhash(im),
        }


def box_stats(boxes: list[YoloBox]) -> list[dict]:
    """Per-box normalized area and aspect ratio (for per-class distribution aggregates)."""
    return [{"area": b.w * b.h, "aspect": b.w / b.h if b.h > 0 else 0.0} for b in boxes]


def frame_flags(metrics: dict) -> list[str]:
    """Informational quality flags for one frame's metrics (subset of the four flag names)."""
    flags = []
    if metrics["brightness"] < FLAG_DARK_MEAN:
        flags.append("dark")
    if metrics["brightness"] > FLAG_BRIGHT_MEAN:
        flags.append("bright")
    if metrics["contrast"] < FLAG_LOW_CONTRAST_STD:
        flags.append("low_contrast")
    if metrics["blur"] < FLAG_BLUR_VAR:
        flags.append("blurry")
    return flags


def is_profile_current(subset_dir: str | Path) -> bool:
    """True iff the subset's profile stamp matches its manifest and the current parameters."""
    subset = Path(subset_dir)
    return provenance.is_current(
        subset / MANIFEST_FILENAME, subset / PROFILE_DIRNAME / STAMP_FILENAME, PROFILE_PARAMS
    )
