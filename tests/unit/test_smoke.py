"""Trivial import/smoke tests guarding the package skeleton."""

from __future__ import annotations

import importlib


def test_package_imports() -> None:
    mod = importlib.import_module("mlops_cv")
    assert mod.__version__


def test_get_settings_is_cached() -> None:
    from mlops_cv.config.settings import get_settings

    assert get_settings() is get_settings()
