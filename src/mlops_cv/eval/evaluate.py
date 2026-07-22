"""Evaluate a model from MLflow (a registry ref or a run URI) on the data subset and log a
reproducible report to MLflow: P/R/mAP50/mAP50-95 (overall + per merged class), a comparison table,
a pass/fail gate (floors + champion/challenger), a latency stub, and the qualitative artifacts —
the fixed demo-clip and the TP/FP error-analysis crops.

The exit code is ``0`` on a gate pass and ``1`` on a fail (``--exit-zero`` forces ``0`` for
orchestrators that branch on the verdict, not the exit code); the last stdout line is a
machine-readable verdict JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from mlops_cv.config import Settings, get_settings
from mlops_cv.eval import gate as gate_mod
from mlops_cv.eval import report as report_mod
from mlops_cv.eval.report import headline_metrics, percentiles
from mlops_cv.tracking import client
from mlops_cv.training.callbacks import per_class_metrics

if TYPE_CHECKING:
    from mlflow.entities.model_registry import ModelVersion
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

DEMO_CLIPS = Path("configs/demo_clips.yaml")


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    """CLI parser; metric defaults come from ``settings.training``"""
    p = argparse.ArgumentParser(
        description="Evaluate a model on a data subset with MLflow logging."
    )
    p.add_argument(
        "--model",
        required=True,
        help="MLflow URI: models:/name@alias, models:/name/version, or runs:/<id>/path",
    )
    p.add_argument(
        "--data", type=Path, default=None, help="dataset YAML (default: the subset YAML)"
    )
    p.add_argument("--split", default="test", help="dataset split to evaluate")
    p.add_argument("--imgsz", type=int, default=settings.training.imgsz)
    p.add_argument("--device", default=settings.training.device)
    p.add_argument("--batch", type=int, default=8, help="val batch (decrease if OOM)")
    p.add_argument("--latency-runs", type=int, default=50)
    p.add_argument("--min-map", type=float, default=0.0, dest="min_map50_95")
    p.add_argument("--min-map50", type=float, default=0.0, dest="min_map50")
    p.add_argument("--min-precision", type=float, default=0.0)
    p.add_argument("--min-recall", type=float, default=0.0)
    p.add_argument(
        "--min-improvement", type=float, default=0.01, help="challenger margin over the champion"
    )
    p.add_argument("--promote", action="store_true", help="on a gate pass, set the champion alias")
    p.add_argument(
        "--run-id",
        default=None,
        help="log to this existing run (e.g. the training run) instead of creating a new one",
    )
    p.add_argument(
        "--exit-zero",
        action="store_true",
        help="always exit 0 (orchestrators branch on the verdict line, not the exit code)",
    )
    p.add_argument("--no-latency", dest="latency_enabled", action="store_false")
    p.add_argument("--no-demos", dest="demos_enabled", action="store_false")
    p.add_argument("--no-crops", dest="crops_enabled", action="store_false")
    return p


def benchmark_latency(
    model: YOLO,
    images: Sequence[str | Path],
    *,
    warmup: int = 3,
    runs: int = 50,
) -> dict[str, float]:
    """Time single-image ``model.predict`` calls; return P50/P95/mean latency (ms) + count.

    A **stub** (batch 1, includes pre/post-processing) for a quick number in the eval run — the
    rigorous engine/VRAM/P99 benchmark is a later optimisation step. Warmup calls are discarded.
    """
    if not images:
        return {
            "latency/p50_ms": 0.0,
            "latency/p95_ms": 0.0,
            "latency/mean_ms": 0.0,
            "latency/n": 0.0,
        }
    paths = [str(p) for p in images]
    timings: list[float] = []
    for i in range(warmup + runs):
        src = paths[i % len(paths)]
        start = time.perf_counter()
        model.predict(src, verbose=False)
        if i >= warmup:
            timings.append((time.perf_counter() - start) * 1000.0)
    pct = percentiles(timings, (50.0, 95.0))
    mean = sum(timings) / len(timings) if timings else 0.0
    return {
        "latency/p50_ms": pct[50.0],
        "latency/p95_ms": pct[95.0],
        "latency/mean_ms": mean,
        "latency/n": float(len(timings)),
    }


def _slug(text: str) -> str:
    """A filesystem/run-name-safe slug for a model reference."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def _resolve_model(model_ref: str) -> tuple[Path, str]:
    """Download a model's ``best.pt`` from MLflow → ``(local path, run-name slug)``.

    ``model_ref`` is an MLflow URI: a run URI ``runs:/<id>/<path>`` or a registry ref
    ``models:/<name>@<alias>`` / ``models:/<name>/<version>``. A registry ref is redirected to its
    version's ``source`` artifact URI and downloaded from there.
    """
    from mlflow.artifacts import download_artifacts

    if model_ref.startswith("models:/"):
        from mlflow import MlflowClient

        spec = model_ref.removeprefix("models:/")
        registry = MlflowClient()
        version: ModelVersion = (
            registry.get_model_version_by_alias(*spec.split("@", 1))
            if "@" in spec
            else registry.get_model_version(*spec.rsplit("/", 1))
        )
        model_ref, slug = version.source, _slug(spec)  # type: ignore[union-attr]
    else:
        slug = _slug(model_ref.split(":/", 1)[1])
    return Path(download_artifacts(artifact_uri=model_ref)), slug


def _promotion_version(model_ref: str, registered_model: str) -> str | None:
    """The registered version to alias for ``--promote``: parsed from a ``models:/`` URI or found by
    run id for a ``runs:/`` URI. A bare local path has no registered version -> ``None``."""
    from mlflow import MlflowClient

    registry = MlflowClient()
    if model_ref.startswith("models:/"):
        spec = model_ref.removeprefix("models:/")
        if "@" in spec:
            name, alias = spec.split("@", 1)
            return registry.get_model_version_by_alias(name, alias).version
        if "/" in spec:
            return spec.rsplit("/", 1)[1]
    if model_ref.startswith("runs:/"):
        run_id = model_ref.removeprefix("runs:/").split("/", 1)[0]
        found = registry.search_model_versions(f"run_id='{run_id}' and name='{registered_model}'")
        if found:
            return found[0].version
    return None


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    args = build_parser(settings).parse_args(argv)

    client.configure()  # export MLFLOW_TRACKING_URI before importing mlflow

    import mlflow
    from ultralytics import YOLO

    mlflow.set_tracking_uri(client.tracking_uri())
    mlflow.set_experiment(settings.mlflow.experiment)

    data_yaml = (args.data or settings.data.subset_dir / settings.data.dataset_yaml.name).resolve()
    weights, model_ref = _resolve_model(args.model)
    images_dir = settings.data.subset_dir / "images" / args.split
    primary = f"{args.split}/mAP50-95"

    logger.info("evaluating %s on %s[%s]", model_ref, data_yaml, args.split)

    run_kwargs = {"run_id": args.run_id} if args.run_id else {"run_name": f"eval-{model_ref}"}
    with mlflow.start_run(**run_kwargs):  # type: ignore[arg-type]
        mlflow.autolog(disable=True)
        mlflow.set_tags({"eval.model": model_ref, "eval.split": args.split})
        model = YOLO(str(weights))

        results = model.val(
            data=str(data_yaml),
            split=args.split,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
        )
        candidate = headline_metrics(results.box, prefix=args.split)
        candidate |= per_class_metrics(
            results.maps, results.ap_class_index, results.names, prefix=primary
        )
        mlflow.log_metrics(candidate)

        champion = gate_mod.fetch_champion_metrics(
            settings.mlflow.registered_model, settings.mlflow.champion_alias
        )
        thresholds = gate_mod.GateThresholds(
            min_map50_95=args.min_map50_95,
            min_map50=args.min_map50,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
            min_improvement=args.min_improvement,
        )
        gate = gate_mod.evaluate_gate(
            candidate, thresholds, champion, primary=primary, prefix=args.split
        )
        mlflow.set_tags({"gate.passed": gate.passed, "gate.challenger_win": gate.is_challenger_win})
        mlflow.log_metric("gate/passed", float(gate.passed))

        latency = None
        if args.latency_enabled:
            latency = benchmark_latency(
                model, sorted(images_dir.glob("*.jpg")), runs=args.latency_runs
            )
            mlflow.log_metrics(latency)

        work_dir = Path(tempfile.mkdtemp(prefix="eval-"))

        if args.demos_enabled:
            try:
                from mlops_cv.data.manifest import DEMO_DIRNAME
                from mlops_cv.eval.visualize import load_demo_clips, render_demo_clips

                # Frames + GT come from the subset's demo store (built by the ingest pipeline).
                videos = render_demo_clips(
                    model,
                    load_demo_clips(DEMO_CLIPS),
                    settings.data.subset_dir / DEMO_DIRNAME,
                    work_dir / "demo",
                )
                for video in videos:
                    mlflow.log_artifact(str(video), artifact_path="demo")
                logger.info("logged %d demo videos", len(videos))
            except Exception as exc:
                logger.warning("demo rendering failed: %s", exc)

        if args.crops_enabled:
            try:
                from mlops_cv.eval.error_analysis import run_error_analysis

                crops = run_error_analysis(
                    model,
                    settings.data.subset_dir / "images" / args.split,
                    settings.data.subset_dir / "labels" / args.split,
                    model.names,
                    work_dir / "error_analysis",
                )
                mlflow.log_artifacts(
                    str(work_dir / "error_analysis"), artifact_path="error_analysis"
                )
                logger.info(
                    "logged %d low-conf TP + %d high-conf FP crops",
                    len(crops.low_conf_tp),
                    len(crops.high_conf_fp),
                )
            except Exception as exc:
                logger.warning("error analysis failed: %s", exc)

        md, csv_path = report_mod.write_report(
            work_dir,
            candidate=candidate,
            champion=champion,
            gate=gate,
            latency=latency,
        )

        mlflow.log_artifact(str(md), artifact_path="eval")
        mlflow.log_artifact(str(csv_path), artifact_path="eval")

        if args.promote and gate.passed:
            version = _promotion_version(args.model, settings.mlflow.registered_model)
            if version is None:
                logger.warning("not promoting: %s has no registered model version", args.model)
            else:
                gate_mod.promote(
                    settings.mlflow.registered_model, version, settings.mlflow.champion_alias
                )
                logger.info("promoted v%s -> alias '%s'", version, settings.mlflow.champion_alias)
        elif args.promote:
            logger.warning("not promoting: gate did not pass")

        logger.info(gate.summary())

    # Machine-readable verdict: orchestrators read the last stdout line (DockerOperator XCom).
    print(
        json.dumps(
            {
                "passed": gate.passed,
                "candidate_primary": gate.candidate_primary,
                "champion_primary": gate.champion_primary,
            }
        ),
        flush=True,
    )
    return 0 if args.exit_zero or gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
