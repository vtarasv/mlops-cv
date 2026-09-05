"""The owned MLflow session: client configuration and the connection.

The MLflow server (started with ``--serve-artifacts``) proxies artifact storage, so a client
only needs ``MLFLOW_TRACKING_URI`` to log runs, params, metrics, and register models.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from mlops_cv.config import Settings, get_settings

if TYPE_CHECKING:
    from types import ModuleType

TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"


def _settings(settings: Settings | None) -> Settings:
    return settings if settings is not None else get_settings()


def configure(settings: Settings | None = None) -> None:
    """Export the tracking URI into ``os.environ`` so ``import mlflow`` reads it.

    Only ``MLFLOW_TRACKING_URI`` — proxied artifacts mean no S3 endpoint or AWS credentials
    are required client-side.
    """
    os.environ[TRACKING_URI_ENV] = _settings(settings).mlflow.tracking_uri


def champion_uri(settings: Settings | None = None) -> str:
    """The registry ref consumers serve by default: ``models:/<registered model>@<alias>``."""
    mlflow_settings = _settings(settings).mlflow
    return f"models:/{mlflow_settings.registered_model}@{mlflow_settings.champion_alias}"


def connect(settings: Settings | None = None, *, experiment: bool = True) -> ModuleType:
    """The configured ``mlflow`` module, ready to open runs."""
    resolved = _settings(settings)
    configure(resolved)

    import mlflow

    mlflow.set_tracking_uri(resolved.mlflow.tracking_uri)
    if experiment:
        mlflow.set_experiment(resolved.mlflow.experiment)
    return mlflow


def registry(settings: Settings | None = None, *, injected: Any | None = None) -> Any:
    """The registry client: ``injected`` if given (tests), else the real one.

    The real client reads the tracking URI from the environment when it is built, so the session
    is configured first — every walk over the registry goes through here and cannot skip that.
    """
    if injected is not None:
        return injected
    configure(settings)
    from mlflow import MlflowClient

    return MlflowClient()
