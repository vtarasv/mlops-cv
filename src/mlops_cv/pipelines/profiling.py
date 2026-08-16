"""Per-frame CV quality metrics for dataset profiling.

Flag thresholds are **informational**: they mark frames in the quality report for a human (or a
future drift monitor) to look at — only corrupt/unreadable images fail the profiling run.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat
from pydantic import BaseModel, ConfigDict, field_validator

from mlops_cv.data.convert_visdrone_vid import YoloBox
from mlops_cv.data.subset import MANIFEST_FILENAME
from mlops_cv.pipelines import provenance

PROFILE_DIRNAME = "profile"
PROFILE_JSON = "profile.json"
QUALITY_REPORT = "quality_report.csv"
DRIFT_BASELINE_CSV = "drift_baseline.csv"
DUPLICATES_CSV = "duplicates.csv"
STAMP_FILENAME = ".profile-stamp.json"

DHASH_SIZE = 8  # 8x8 difference hash -> 64-bit hex string

DRIFT_METRICS = ("brightness", "contrast", "blur")
BASELINE_SPLIT = "train"

BASELINE_FIELDS = ["sequence", "n_frames", *DRIFT_METRICS]
DUPLICATES_FIELDS = ["dhash", "image_relpath"]

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
    "baseline_split": BASELINE_SPLIT,
    "baseline_metrics": list(DRIFT_METRICS),
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


class BaselineScene(BaseModel):
    """One training scene of the drift baseline: its frame count and its metric means.
    A drift monitor scores a live window against the spread of these means.
    """

    model_config = ConfigDict(frozen=True)

    sequence: str
    n_frames: int
    means: dict[str, float]

    @field_validator("means")
    @classmethod
    def _means_cover_the_drift_metrics(cls, means: dict[str, float]) -> dict[str, float]:
        if set(means) != set(DRIFT_METRICS):
            raise ValueError(f"means {sorted(means)} must be keyed by {list(DRIFT_METRICS)}")
        return means


def baseline_header() -> str:
    """The drift baseline's CSV header line."""
    return ",".join(BASELINE_FIELDS)


def baseline_csv_line(scene: BaselineScene) -> str:
    """One scene as a CSV line (no trailing newline), field order = ``BASELINE_FIELDS``."""
    row: dict[str, object] = {"sequence": scene.sequence, "n_frames": scene.n_frames, **scene.means}
    buf = io.StringIO()
    csv.writer(buf).writerow([row[field] for field in BASELINE_FIELDS])
    return buf.getvalue().rstrip("\r\n")


def baseline_rows(lines: Iterable[str]) -> list[BaselineScene]:
    """Parse drift-baseline CSV lines into scenes, validating the header."""
    reader = csv.DictReader(lines)
    if reader.fieldnames is None or list(reader.fieldnames) != BASELINE_FIELDS:
        raise ValueError(
            f"unexpected drift baseline header {reader.fieldnames} (expected {BASELINE_FIELDS})"
        )
    return [
        BaselineScene(
            sequence=row["sequence"],
            n_frames=int(row["n_frames"]),
            means={name: float(row[name]) for name in DRIFT_METRICS},
        )
        for row in reader
    ]


def read_baseline(profile_dir: str | Path) -> list[BaselineScene]:
    """Read a profile directory's drift baseline — one scene per training sequence."""
    with (Path(profile_dir) / DRIFT_BASELINE_CSV).open(newline="", encoding="utf-8") as fh:
        return baseline_rows(fh)


def profile_counts(profile: dict) -> dict[str, float]:
    """A Profile's headline counts — the fixed-size readings that describe a data version."""
    return {
        **profile["dataset"],
        "n_baseline_sequences": profile["baseline"]["n_sequences"],
    }


def profile_identity(subset_dir: str | Path) -> str:
    """The provenance stamp identifying the Profile of this subset's *current* data."""
    subset = Path(subset_dir)
    return provenance.stamp_payload(
        provenance.manifest_sha256(subset / MANIFEST_FILENAME), PROFILE_PARAMS
    )


def is_profile_current(subset_dir: str | Path) -> bool:
    """True iff the subset's profile stamp matches its manifest and the current parameters."""
    subset = Path(subset_dir)
    return provenance.is_current(
        subset / MANIFEST_FILENAME, subset / PROFILE_DIRNAME / STAMP_FILENAME, PROFILE_PARAMS
    )
