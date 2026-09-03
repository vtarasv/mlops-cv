"""Train YOLO26s on the VisDrone-VID subset, log to MLflow, and register the model.

Training is pure model production (train + per-epoch val); the held-out test metrics, report,
gate, and qualitative artifacts are the evaluation harness's job
(``python -m mlops_cv.evaluation --run-id``).
The last stdout line is a machine-readable JSON for orchestrators.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from mlops_cv.config import Settings, get_settings
from mlops_cv.orchestration.handoff import BEST_WEIGHTS_RELPATH, TrainHandoff
from mlops_cv.tracking import client, data_version
from mlops_cv.training.callbacks import make_batch_params_callback, make_per_class_callback

if TYPE_CHECKING:
    from types import ModuleType

    from mlflow import ActiveRun

logger = logging.getLogger(__name__)


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    """CLI parser whose defaults come from ``settings.training``."""
    t = settings.training
    p = argparse.ArgumentParser(description="Train YOLO26s on a dataset with MLflow logging.")
    p.add_argument("--epochs", type=int, default=t.epochs)
    p.add_argument("--imgsz", type=int, default=t.imgsz)
    p.add_argument("--batch", type=int, default=t.batch, help="-1 = ultralytics autobatch")
    p.add_argument("--device", default=t.device)
    p.add_argument("--no-amp", dest="amp", action="store_false", help="disable AMP (fp32 fallback)")
    p.set_defaults(amp=t.amp)
    return p


def _log_dataset(mlflow: ModuleType, manifest: Path) -> None:
    """Log the dataset manifest as an MLflow input + a content hash tag for lineage."""
    import pandas as pd

    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with warnings.catch_warnings():
        # mlflow.data emits benign hints here
        warnings.simplefilter("ignore", UserWarning)
        dataset = mlflow.data.from_pandas(
            pd.read_csv(manifest), source=manifest.resolve().as_uri(), name=manifest.parent.name
        )
        mlflow.log_input(dataset, context="training")
    mlflow.set_tag("dataset_sha", sha)


def _log_data_version(mlflow: ModuleType, settings: Settings) -> None:
    """Link the run to the data version describing what it is training on."""
    try:
        run_id = data_version.published_run(settings.data.subset_dir, settings, mlflow=mlflow)
    except Exception as exc:  # any lookup failure: the model still gets produced and registered
        logger.warning(f"data-version lookup failed ({exc}) — training continues without the link")
        return
    if run_id is None:
        logger.warning(
            f"no published data version describes {settings.data.subset_dir} — run "
            "`make profile` to publish one; training continues without a data-version link"
        )
        return
    mlflow.set_tag(data_version.RUN_TAG, run_id)
    logger.info(f"training data version: {run_id}")


def _register_model(run: ActiveRun, name: str) -> str:
    """Register the val-selected ``best.pt`` as a new model version; return its version number.

    ``weights/best.pt`` is already logged by ultralytics' built-in callback.
    """
    from mlflow import MlflowClient
    from mlflow.exceptions import RestException

    registry = MlflowClient()
    with contextlib.suppress(RestException):
        registry.create_registered_model(name)  # ok if it already exists
    version = registry.create_model_version(
        name=name, source=f"{run.info.artifact_uri}/{BEST_WEIGHTS_RELPATH}", run_id=run.info.run_id
    )
    return version.version


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    args = build_parser(settings).parse_args(argv)

    mlflow = client.connect(settings)

    from ultralytics import YOLO
    from ultralytics import settings as yolo_settings

    t = settings.training
    data_yaml = (settings.data.subset_dir / settings.data.dataset_yaml.name).resolve()

    yolo_settings.update({"mlflow": True})  # enable the built-in MLflow callback
    os.environ["MLFLOW_EXPERIMENT_NAME"] = settings.mlflow.experiment

    run_name = f"{Path(t.weights).stem}-imgsz{args.imgsz}-e{args.epochs}"
    logger.info(f"training {t.weights} on {data_yaml} -> MLflow {client.tracking_uri()}")

    with mlflow.start_run(run_name=run_name) as run:
        model = YOLO(t.weights)
        model.add_callback("on_train_start", make_batch_params_callback())
        model.add_callback("on_fit_epoch_end", make_per_class_callback())
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            nbs=t.nbs,
            amp=args.amp,
            cache=t.cache,
            workers=t.workers,
            device=args.device,
            patience=t.patience,
        )
        mlflow.autolog(disable=True)  # the built-in callback enabled autolog
        _log_dataset(mlflow, settings.data.subset_dir / "manifest.csv")
        _log_data_version(mlflow, settings)
        version = _register_model(run, settings.mlflow.registered_model)
        run_id = run.info.run_id

    # Machine-readable handoff: orchestrators read the last stdout line (DockerOperator XCom).
    print(TrainHandoff(version=version, run_id=run_id).to_line(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
