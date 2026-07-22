"""Provenance stamps: skip-if-current checks for derived artifacts of a dataset manifest.

A pipeline writes a stamp (source-manifest hash + its parameters) as its LAST step, so a partial
run is never considered current. Orchestration branches call :func:`is_current` to skip work
whose inputs and parameters are unchanged. Pure stdlib — importable by the Airflow scheduler.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def manifest_sha256(manifest_path: str | Path) -> str:
    """Content hash of a dataset manifest — the provenance key for artifacts derived from it."""
    return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()


def stamp_payload(manifest_sha: str, params: dict) -> str:
    """Canonical JSON recording what an artifact was built from (source hash + parameters)."""
    return json.dumps({"source_manifest_sha256": manifest_sha, "params": params}, sort_keys=True)


def is_current(manifest_path: str | Path, stamp_path: str | Path, params: dict) -> bool:
    """True iff ``stamp_path`` matches ``manifest_path``'s content and ``params``.

    False on any missing manifest/stamp or unreadable stamp — callers treat that as "stale,
    rebuild".
    """
    manifest, stamp = Path(manifest_path), Path(stamp_path)
    if not manifest.is_file() or not stamp.is_file():
        return False
    expected = stamp_payload(manifest_sha256(manifest), params)
    try:
        return json.loads(stamp.read_text(encoding="utf-8")) == json.loads(expected)
    except json.JSONDecodeError:
        return False
