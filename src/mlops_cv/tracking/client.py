"""The owned MLflow session: client configuration and the connection.

The MLflow server (started with ``--serve-artifacts``) proxies artifact storage, so a client
only needs ``MLFLOW_TRACKING_URI`` to log runs, params, metrics, and register models.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from mlops_cv.config import Settings, get_settings

if TYPE_CHECKING:
    from types import ModuleType

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


def champion_uri(settings: Settings | None = None) -> str:
    """The registry ref consumers serve by default: ``models:/<registered model>@<alias>``."""
    mlflow_settings = _settings(settings).mlflow
    return f"models:/{mlflow_settings.registered_model}@{mlflow_settings.champion_alias}"


def connect(settings: Settings | None = None, *, experiment: bool = True) -> ModuleType:
    """The configured ``mlflow`` module, ready to open runs."""
    resolved = _settings(settings)
    configure(resolved)

    import mlflow

    mlflow.set_tracking_uri(tracking_uri(resolved))
    if experiment:
        mlflow.set_experiment(resolved.mlflow.experiment)
    return mlflow
