"""Custom ultralytics training callbacks that log extra detail to the active MLflow run."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ultralytics.engine.trainer import BaseTrainer

logger = logging.getLogger(__name__)


def per_class_metrics(
    maps: Sequence[float] | NDArray,
    ap_class_index: Sequence[int] | NDArray,
    names: Mapping[int, str],
    *,
    prefix: str = "metrics/mAP50-95",
) -> dict[str, float]:
    """Map per-class mAP50-95 to ``{prefix}/<class-name>`` for the classes actually evaluated."""
    return {f"{prefix}/{names[int(c)]}": float(maps[int(c)]) for c in ap_class_index}


def make_per_class_callback() -> Callable[[BaseTrainer], None]:
    """An ``on_fit_epoch_end`` callback logging per-merged-class val mAP50-95 to the active run."""

    def _log_per_class(trainer: BaseTrainer) -> None:
        import mlflow

        if mlflow.active_run() is None:
            return
        try:
            metrics = trainer.validator.metrics  # type: ignore[union-attr]
            values = per_class_metrics(metrics.maps, metrics.ap_class_index, metrics.names)
            mlflow.log_metrics(values, step=int(trainer.epoch))
        except Exception as exc:
            logger.warning("per-class metric logging failed: %s", exc)

    return _log_per_class


def make_batch_params_callback() -> Callable[[BaseTrainer], None]:
    """An ``on_train_start`` callback logging the autobatch-resolved batch + grad accumulation."""

    def _log_batch_params(trainer: BaseTrainer) -> None:
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
            logger.warning("batch-params logging failed: %s", exc)

    return _log_batch_params
