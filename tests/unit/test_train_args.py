"""Unit tests for the training CLI parser (no GPU, no mlflow; defaults flow from Settings)."""

from __future__ import annotations

import pytest

from mlops_cv.config import load_settings
from mlops_cv.training.train import TARGET_MODELS, build_parser


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


def test_trial_flags() -> None:
    args = build_parser(_settings()).parse_args(
        ["--model", "yolo26n", "--seed", "2", "--set", "scale=[0.5,3.0]", "--no-register"]
    )
    assert (args.model, args.seed, args.register) == ("yolo26n", 2, False)
    assert args.overrides == [("scale", [0.5, 3.0])]


def test_production_defaults_register_into_the_server_target() -> None:
    args = build_parser(_settings()).parse_args([])
    assert (args.target, args.model, args.register, args.overrides) == ("server", None, True, [])
    assert args.imgsz == 1280


def test_each_target_trains_its_production_default_size() -> None:
    assert TARGET_MODELS == {"server": "yolo26m", "edge": "yolo26n"}


def test_set_needs_key_equals_value() -> None:
    with pytest.raises(SystemExit):
        build_parser(_settings()).parse_args(["--set", "scale"])
