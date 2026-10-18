"""The Mac-lock metadata for tensorrt-cu13-libs follows the tensorrt-cu13 pin (docs/macos.md)."""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())


def test_tensorrt_libs_metadata_follows_the_tensorrt_pin() -> None:
    pin = next(
        d for d in PYPROJECT["dependency-groups"]["optimize"] if d.startswith("tensorrt-cu13==")
    )
    (meta,) = [
        m
        for m in PYPROJECT["tool"]["uv"]["dependency-metadata"]
        if m["name"] == "tensorrt-cu13-libs"
    ]
    assert pin.split("==")[1].split(";")[0].strip() == meta["version"]
