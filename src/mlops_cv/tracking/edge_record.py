"""What the edge bench leaves on the record run for the desktop gate, and what the gate sets.

The device logs a detections dump and its package-version tags; the desktop reads both back and,
on a pass, sets the measured tag. Both halves build these names and payloads only through here.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from mlops_cv.tracking.metric_keys import OPTIMIZE_PREFIX


def detections_path(label: str, variant: str) -> str:
    """The run-artifact path of one variant's dump from one device."""
    return f"edge/{label}/{variant}/detections.json"


def report_path(label: str) -> str:
    """The run-artifact path of the gate's per-device report."""
    return f"edge/{label}.md"


def to_payload(detections: Mapping[str, np.ndarray]) -> dict[str, list[list[float]]]:
    """``{frame file name: (n, 6) rows}`` -> the JSON-ready dump, full float32 precision."""
    return {name: np.asarray(rows, np.float32).tolist() for name, rows in detections.items()}


def from_payload(payload: Mapping[str, list[list[float]]]) -> dict[str, np.ndarray]:
    """The dump read back: ``{frame file name: (n, 6) x1 y1 x2 y2 conf cls}``, source pixels."""
    return {name: np.asarray(rows, np.float32).reshape(-1, 6) for name, rows in payload.items()}


def package_tag(label: str, package: str) -> str:
    """The run tag holding one device package's version: ``optimize.<label>.<package>``."""
    return f"{OPTIMIZE_PREFIX}.{label}.{package}"


def package_versions(tags: Mapping[str, str], label: str) -> dict[str, str]:
    """The device's package versions, read back from the run's tags."""
    prefix = package_tag(label, "")
    return {key.removeprefix(prefix): v for key, v in tags.items() if key.startswith(prefix)}


def measured_tag(label: str) -> str:
    """Set by the desktop gate once every variant's device reading is within tolerance."""
    return f"{OPTIMIZE_PREFIX}.measured.{label}"
