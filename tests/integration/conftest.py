"""Shared fixtures for the MLflow integration tests (need the stack up + the ``mlflow`` client)."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from mlops_cv.config import get_settings
from mlops_cv.tracking import client

_TIMEOUT = 5


@pytest.fixture(scope="module")
def registry():
    """A configured ``MlflowClient`` pointed at the live stack.

    Skips the test when ``mlflow`` is absent (CI's ``--no-group gpu``) or the stack isn't reachable.
    """
    mlflow = pytest.importorskip("mlflow")
    client.configure()
    uri = get_settings().mlflow.tracking_uri
    try:
        urllib.request.urlopen(f"{uri}/health", timeout=_TIMEOUT)  # noqa: S310 (localhost only)
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"MLflow stack not reachable at {uri}: {exc}")
    mlflow.set_tracking_uri(uri)
    from mlflow import MlflowClient

    return MlflowClient()
