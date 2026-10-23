"""The third-party checkpoint download: a cached file is reused only if its hash matches."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mlops_cv.evaluation.third_party import _download
from mlops_cv.startup import StartupError


def test_a_cached_checkpoint_with_the_pinned_hash_is_reused(tmp_path: Path) -> None:
    dest = tmp_path / "best.pt"
    dest.write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest()
    assert _download("org/repo", "rev", sha, dest) == dest


def test_a_cached_checkpoint_with_another_hash_fails_naming_the_fix(tmp_path: Path) -> None:
    dest = tmp_path / "best.pt"
    dest.write_bytes(b"tampered")
    with pytest.raises(StartupError, match="delete it to re-download"):
        _download("org/repo", "rev", "0" * 64, dest)
