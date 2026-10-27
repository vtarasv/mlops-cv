"""Edge-image Python recipes that mirror a uv.lock package must pin the lock's version."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECIPE_NAME = re.compile(r"python3-(?P<pkg>.+?)(?:_(?P<ver>[^_]+))?\.bb")
PV_LINE = re.compile(r'^PV\s*=\s*"([^"]+)"', re.MULTILINE)


def _lock() -> dict[str, str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {p["name"]: p["version"] for p in lock["package"] if "version" in p}


def _recipes() -> dict[str, str | None]:
    found = {}
    for bb in (ROOT / "yocto" / "meta-mlops-cv").rglob("python3-*.bb"):
        match = RECIPE_NAME.fullmatch(bb.name)
        if match is None:
            continue
        pv = PV_LINE.search(bb.read_text(encoding="utf-8"))
        pkg = re.sub(r"[-_.]+", "-", match["pkg"]).lower()
        found[pkg] = pv[1] if pv else match["ver"]
    return found


def test_every_recipe_mirroring_a_lock_package_pins_the_lock_version() -> None:
    lock = _lock()
    pinned = {pkg: ver for pkg, ver in _recipes().items() if pkg in lock}
    assert pinned, "no meta-mlops-cv recipe maps to a uv.lock package: the glob went stale"
    assert pinned == {pkg: lock[pkg] for pkg in pinned}
