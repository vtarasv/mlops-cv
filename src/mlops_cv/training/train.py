"""Train YOLO26s on the VisDrone-VID subset, log to MLflow, and register the model."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import logging
import os
import warnings
from pathlib import Path

from mlops_cv.config import Settings, get_settings
from mlops_cv.tracking import client
from mlops_cv.training.callbacks import (
    make_batch_params_callback,
    make_per_class_callback,
    per_class_metrics,
)

logger = logging.getLogger(__name__)

DEMO_CLIPS = Path("configs/demo_clips.yaml")


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    """CLI parser whose defaults come from ``settings.training``."""
    t = settings.training
    p = argparse.ArgumentParser(description="Train YOLO26s on VisDrone-VID with MLflow logging.")
    p.add_argument("--epochs", type=int, default=t.epochs)
    p.add_argument("--imgsz", type=int, default=t.imgsz)
    p.add_argument("--batch", type=int, default=t.batch, help="-1 = ultralytics autobatch")
    p.add_argument("--device", default=t.device)
    p.add_argument(
        "--data", type=Path, default=None, help="dataset YAML (default: the generated subset YAML)"
    )
    p.add_argument("--no-amp", dest="amp", action="store_false", help="disable AMP (fp32 fallback)")
    p.set_defaults(amp=t.amp)
    return p


def _log_test_metrics(mlflow, results) -> None:
    """Log the held-out test-dev metrics as the run's headline (overall + per merged class)."""
    box = results.box
    mlflow.log_metrics(
        {
            "test/precision": float(box.mp),
            "test/recall": float(box.mr),
            "test/mAP50": float(box.map50),
            "test/mAP50-95": float(box.map),
        }
    )
    mlflow.log_metrics(
        per_class_metrics(
            results.maps, results.ap_class_index, results.names, prefix="test/mAP50-95"
        )
    )


def _log_dataset(mlflow, manifest: Path) -> None:
    """Log the dataset manifest as an MLflow input + a content hash tag for lineage."""
    import pandas as pd

    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with warnings.catch_warnings():
        # mlflow.data emits benign hints here
        warnings.simplefilter("ignore", UserWarning)
        dataset = mlflow.data.from_pandas(
            pd.read_csv(manifest), source=manifest.resolve().as_uri(), name="visdrone-vid-small"
        )
        mlflow.log_input(dataset, context="training")
    mlflow.set_tag("dataset_sha", sha)


def _register_model(run, name: str) -> None:
    """Register the val-selected ``best.pt`` as a new model version.

    ``weights/best.pt`` is already logged by ultralytics' built-in callback.
    """
    from mlflow import MlflowClient
    from mlflow.exceptions import RestException

    registry = MlflowClient()
    with contextlib.suppress(RestException):
        registry.create_registered_model(name)  # ok if it already exists
    registry.create_model_version(
        name=name, source=f"{run.info.artifact_uri}/weights/best.pt", run_id=run.info.run_id
    )


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    args = build_parser(settings).parse_args(argv)

    client.configure()  # export MLFLOW_TRACKING_URI so the ultralytics callback hits our server

    import mlflow
    import torch
    from ultralytics import YOLO
    from ultralytics import settings as yolo_settings

    from mlops_cv.eval.visualize import load_demo_clips, render_demo_clips

    t = settings.training
    data_yaml = (args.data or settings.data.subset_dir / settings.data.dataset_yaml.name).resolve()

    yolo_settings.update({"mlflow": True})  # enable the built-in MLflow callback
    os.environ["MLFLOW_EXPERIMENT_NAME"] = settings.mlflow.experiment
    mlflow.set_tracking_uri(client.tracking_uri())
    mlflow.set_experiment(settings.mlflow.experiment)

    run_name = f"{Path(t.weights).stem}-imgsz{args.imgsz}-e{args.epochs}"
    logger.info("training %s on %s -> MLflow %s", t.weights, data_yaml, client.tracking_uri())

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
        best = Path(model.trainer.best)  # type: ignore
        save_dir = Path(model.trainer.save_dir)  # type: ignore
        test_batch = model.trainer.batch_size * 2  # type: ignore
        del model
        torch.cuda.empty_cache()

        # Headline metrics on the held-out test split (best.pt was selected on val).
        evaluator = YOLO(str(best))
        _log_test_metrics(
            mlflow,
            evaluator.val(
                data=str(data_yaml),
                split="test",
                imgsz=args.imgsz,
                device=args.device,
                batch=test_batch,
            ),
        )

        _log_dataset(mlflow, settings.data.subset_dir / "manifest.csv")
        _register_model(run, settings.mlflow.registered_model)

        try:
            videos = render_demo_clips(
                evaluator, load_demo_clips(DEMO_CLIPS), settings.data.raw_dir, save_dir / "demo"
            )
            for video in videos:
                mlflow.log_artifact(str(video), artifact_path="demo")
            logger.info("logged %d demo videos", len(videos))
        except Exception as exc:
            logger.warning("demo rendering failed: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
