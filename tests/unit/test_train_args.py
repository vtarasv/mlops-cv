"""Unit tests for the training CLI parser (no GPU, no mlflow; defaults flow from Settings)."""

from __future__ import annotations

from mlops_cv.config import load_settings
from mlops_cv.training.train import build_parser


def _settings():
    return load_settings(base_dir="/nonexistent")  # model defaults, no env files


def test_defaults_from_settings() -> None:
    s = _settings()
    args = build_parser(s).parse_args([])
    assert args.epochs == s.training.epochs
    assert args.imgsz == s.training.imgsz
    assert args.batch == s.training.batch
    assert args.device == s.training.device
    assert args.amp is True


def test_flag_overrides() -> None:
    args = build_parser(_settings()).parse_args(["--epochs", "5", "--imgsz", "320", "--no-amp"])
    assert args.epochs == 5
    assert args.imgsz == 320
    assert args.amp is False


def test_batch_override() -> None:
    args = build_parser(_settings()).parse_args(["--batch", "8"])
    assert args.batch == 8
