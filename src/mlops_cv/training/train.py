"""Train YOLO26 on the subset, log to MLflow, and register the model (or record a trial).

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
from typing import TYPE_CHECKING, Any

import yaml

from mlops_cv.config import Settings, get_settings
from mlops_cv.data.subset import dataset_yaml
from mlops_cv.orchestration.handoff import BEST_WEIGHTS_RELPATH, TrainHandoff
from mlops_cv.tracking import client, data_version
from mlops_cv.training.callbacks import log_batch_params, log_per_class

if TYPE_CHECKING:
    from types import ModuleType

    from mlflow import ActiveRun

logger = logging.getLogger(__name__)

MODELS = ("yolo26n", "yolo26s", "yolo26m")

# A trial's final val and latency both read this split.
TRIAL_SPLIT = "val"


def _override(text: str) -> tuple[str, Any]:
    """``key=value`` -> ``(key, YAML-parsed value)``: ``scale=[0.5,3.0]`` arrives as a list."""
    key, sep, value = text.partition("=")
    if not (key and sep):
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    return key, yaml.safe_load(value)


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    """CLI parser whose defaults come from ``settings.training``."""
    t = settings.training
    p = argparse.ArgumentParser(description="Train YOLO26 on a dataset with MLflow logging.")
    p.add_argument("--model", choices=MODELS, default=Path(t.weights).stem)
    p.add_argument("--epochs", type=int, default=t.epochs)
    p.add_argument("--imgsz", type=int, default=t.imgsz)
    p.add_argument("--batch", type=int, default=t.batch, help="-1 = ultralytics autobatch")
    p.add_argument("--device", default=t.device)
    p.add_argument("--no-amp", dest="amp", action="store_false", help="disable AMP (fp32 fallback)")
    p.set_defaults(amp=t.amp)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--set",
        dest="overrides",
        type=_override,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra model.train() argument, value parsed as YAML (repeatable; wins over flags)",
    )
    p.add_argument("--run-name", default=None)
    p.add_argument(
        "--no-register",
        dest="register",
        action="store_false",
        help="a trial: record val metrics on the run instead of registering a model version",
    )
    return p


def _log_dataset(mlflow: ModuleType, manifest: Path) -> None:
    """Log the dataset manifest as an MLflow input (name + source + content digest) for lineage."""
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.data.meta_dataset import MetaDataset

    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with warnings.catch_warnings():
        # the registry warns that a local path matches two equivalent source classes
        warnings.simplefilter("ignore", UserWarning)
        source = resolve_dataset_source(str(manifest.resolve()))
    # The server caps a digest at 36 chars; the full hash goes on the tag below.
    dataset = MetaDataset(source, name=manifest.parent.name, digest=sha[:16])
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
    data_yaml = dataset_yaml(settings)

    yolo_settings.update({"mlflow": True})  # enable the built-in MLflow callback
    os.environ["MLFLOW_EXPERIMENT_NAME"] = settings.mlflow.experiment

    weights = Path(t.weights).with_name(f"{args.model}.pt")
    run_name = args.run_name or f"{args.model}-imgsz{args.imgsz}-e{args.epochs}"
    logger.info(f"training {weights} on {data_yaml} -> MLflow {settings.mlflow.tracking_uri}")

    with mlflow.start_run(run_name=run_name) as run:
        model = YOLO(weights)
        model.add_callback("on_train_start", log_batch_params)
        model.add_callback("on_fit_epoch_end", log_per_class)
        train_args = dict(
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
            seed=args.seed,
            name=run_name,
        ) | dict(args.overrides)
        model.train(**train_args)
        mlflow.autolog(disable=True)  # the built-in callback enabled autolog
        _log_dataset(mlflow, settings.data.subset_dir / "manifest.csv")
        _log_data_version(mlflow, settings)
        run_id = run.info.run_id
        if not args.register:
            from mlops_cv.evaluation.evaluate import split_metrics

            # model now holds best.pt; latency is the runner's, measured alone after training
            mlflow.log_metrics(
                split_metrics(
                    model,
                    data_yaml,
                    TRIAL_SPLIT,
                    imgsz=int(train_args["imgsz"]),
                    device=args.device,
                    batch=8,
                    name=f"{run_name}-{TRIAL_SPLIT}",
                )
            )
            logger.info(f"trial {run_name} recorded on run {run_id}, not registered")
            return 0
        version = _register_model(run, settings.mlflow.registered_model)

    # Machine-readable handoff: orchestrators read the last stdout line (DockerOperator XCom).
    print(TrainHandoff(version=version, run_id=run_id).to_line(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
