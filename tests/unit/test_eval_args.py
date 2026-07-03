"""Unit tests for the eval CLI parser."""

from __future__ import annotations

import pytest

from mlops_cv.config import load_settings
from mlops_cv.eval.evaluate import build_parser


def _settings():
    return load_settings(base_dir="/nonexistent")  # model defaults, no env files


def test_defaults_from_settings() -> None:
    s = _settings()
    args = build_parser(s).parse_args(["--model", "best.pt"])
    assert args.model == "best.pt"
    assert args.split == "test"
    assert args.imgsz == s.training.imgsz
    assert args.device == s.training.device
    assert args.batch == 8
    assert args.promote is False
    assert args.latency_enabled is True
    assert args.demos_enabled is True
    assert args.crops_enabled is True
    assert args.data is None
    assert args.min_map50_95 == 0.0
    assert args.min_improvement == 0.01
    assert args.run_id is None  # default: a fresh eval run, not a resumed one
    assert args.exit_zero is False


def test_model_required() -> None:
    with pytest.raises(SystemExit):
        build_parser(_settings()).parse_args([])


def test_flag_overrides() -> None:
    args = build_parser(_settings()).parse_args(
        [
            "--model",
            "m",
            "--imgsz",
            "320",
            "--promote",
            "--no-latency",
            "--no-demos",
            "--no-crops",
            "--min-map",
            "0.2",
            "--min-improvement",
            "0.03",
        ]
    )
    assert args.imgsz == 320
    assert args.promote is True
    assert args.latency_enabled is False
    assert args.demos_enabled is False
    assert args.crops_enabled is False
    assert args.min_map50_95 == 0.2
    assert args.min_improvement == 0.03  # overrides the 0.01 default


def test_orchestrator_flags() -> None:
    args = build_parser(_settings()).parse_args(
        ["--model", "runs:/abc/weights/best.pt", "--run-id", "abc", "--exit-zero"]
    )
    assert args.run_id == "abc"  # same-run mode: eval lands on the training run
    assert args.exit_zero is True  # a challenger loss is a verdict, not a failure
