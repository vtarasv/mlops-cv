"""Unit tests for the tuning-stage runner's config and argv (no GPU, no mlflow)."""

from __future__ import annotations

from pathlib import Path

from mlops_cv.config import load_settings
from mlops_cv.training.train import build_parser
from mlops_cv.tuning.common import load_stage
from mlops_cv.tuning.tune import trial_argv

STAGE = """
defaults: {patience: 0, close_mosaic: 10, imgsz: 640}
trials:
  - {name: a, model: yolo26n}
  - {name: b, model: yolo26m, imgsz: 1280, scale: [0.5, 3.0], cache: ram}
"""


def _stage(tmp_path: Path, text: str = STAGE) -> Path:
    path = tmp_path / "stage1.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_knobs_reach_model_train_with_their_types(tmp_path: Path) -> None:
    """Flags stay flags; the rest round-trips through ``--set`` as YAML (a list stays a list)."""
    argv = trial_argv("stage1-b", load_stage(_stage(tmp_path))["stage1-b"])
    assert argv[1:3] == ["-m", "mlops_cv.training"]
    args = build_parser(load_settings(base_dir="/nonexistent")).parse_args(argv[3:])
    assert (args.model, args.imgsz, args.register, args.run_name) == (
        "yolo26m",
        1280,
        False,
        "stage1-b",
    )
    assert dict(args.overrides) == {
        "patience": 0,
        "close_mosaic": 10,
        "scale": [0.5, 3.0],
        "cache": "ram",
    }
