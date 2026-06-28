"""Resolve MLflow client configuration from :class:`Settings`.

The MLflow server (started with ``--serve-artifacts``) proxies artifact storage, so a client
only needs ``MLFLOW_TRACKING_URI`` to log runs, params, metrics, and register models.
"""

from __future__ import annotations

import os

from mlops_cv.config import Settings, get_settings

TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"


def _settings(settings: Settings | None) -> Settings:
    return settings if settings is not None else get_settings()


def tracking_uri(settings: Settings | None = None) -> str:
    """The MLflow tracking-server URI (``settings.mlflow.tracking_uri``)."""
    return _settings(settings).mlflow.tracking_uri


def mlflow_env(settings: Settings | None = None) -> dict[str, str]:
    """The environment a client needs to reach the tracking server.

    Only ``MLFLOW_TRACKING_URI`` — proxied artifacts mean no S3 endpoint or AWS credentials
    are required client-side.
    """
    return {TRACKING_URI_ENV: tracking_uri(settings)}


def configure(settings: Settings | None = None) -> None:
    """Export :func:`mlflow_env` into ``os.environ`` so ``import mlflow`` reads it."""
    os.environ.update(mlflow_env(settings))
