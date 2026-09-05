"""Custom ultralytics training callbacks that log extra detail to the active MLflow run."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mlops_cv.tracking.metric_keys import per_class_metrics

if TYPE_CHECKING:
    from ultralytics.engine.trainer import BaseTrainer

logger = logging.getLogger(__name__)


def log_per_class(trainer: BaseTrainer) -> None:
    """``on_fit_epoch_end``: log per-merged-class val mAP50-95 to the active run."""
    import mlflow

    if mlflow.active_run() is None:
        return
    try:
        metrics = trainer.validator.metrics  # type: ignore[union-attr]
        values = per_class_metrics(metrics.maps, metrics.ap_class_index, metrics.names)
        mlflow.log_metrics(values, step=int(trainer.epoch))
    except Exception as exc:
        logger.warning(f"per-class metric logging failed: {exc}")


def log_batch_params(trainer: BaseTrainer) -> None:
    """``on_train_start``: log the autobatch-resolved batch + grad accumulation."""
    import mlflow

    if mlflow.active_run() is None:
        return
    try:
        batch, accumulate = int(trainer.batch_size), int(trainer.accumulate)
        mlflow.log_params(
            {
                "resolved_batch": batch,
                "accumulate": accumulate,
                "effective_batch": batch * accumulate,
            }
        )
    except Exception as exc:
        logger.warning(f"batch-params logging failed: {exc}")
