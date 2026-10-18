"""Importing mlops_cv points ultralytics at the repo's settings, never the user-global file."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _run(code: str, **env: str) -> str:
    clean = {k: v for k, v in os.environ.items() if k != "YOLO_CONFIG_DIR"}
    cmd = [sys.executable, "-c", code]
    out = subprocess.run(cmd, env={**clean, **env}, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()[-1]  # ultralytics prints a notice when it creates the file


def test_import_defaults_the_config_dir_to_the_repo() -> None:
    code = "import os, mlops_cv; print(os.environ['YOLO_CONFIG_DIR'])"
    assert _run(code) == str(REPO / ".ultralytics")


def test_an_explicit_config_dir_wins() -> None:
    code = "import os, mlops_cv; print(os.environ['YOLO_CONFIG_DIR'])"
    assert _run(code, YOLO_CONFIG_DIR="/elsewhere") == "/elsewhere"


def test_ultralytics_reads_the_repo_settings() -> None:
    pytest.importorskip("ultralytics")
    code = "import mlops_cv; from ultralytics.utils import SETTINGS_FILE; print(SETTINGS_FILE)"
    assert Path(_run(code)).is_relative_to(REPO / ".ultralytics")
