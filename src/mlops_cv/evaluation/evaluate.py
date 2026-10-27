"""Evaluate a model from MLflow (a registry ref or a run URI) on the data subset and log a
reproducible report to MLflow: P/R/mAP50/mAP50-95 (overall + per merged class), mAP50-95_small,
a comparison table, a pass/fail gate (floors + champion/challenger), a latency stub, and the
qualitative artifacts — the fixed demo-clip and the TP/FP error-analysis crops.

The exit code is ``0`` on a gate pass and ``1`` on a fail (``--exit-zero`` forces ``0`` for
orchestrators that branch on the verdict, not the exit code); the last stdout line is a
machine-readable verdict JSON.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Any

from mlops_cv import benchmark as benchmark_mod
from mlops_cv.config import Settings, get_settings
from mlops_cv.data.subset import (
    DEMO_DIRNAME,
    IMAGES_DIRNAME,
    LABELS_DIRNAME,
    dataset_yaml,
    demo_store_present,
)
from mlops_cv.evaluation import ap_small as ap_small_mod
from mlops_cv.evaluation import gate as gate_mod
from mlops_cv.evaluation import report as report_mod
from mlops_cv.evaluation.score_detections import split_labels
from mlops_cv.optimize.variants import (
    HEADS,
    ONE_TO_MANY,
    ONE_TO_ONE,
    SERVER,
    TARGETS,
    exported_head,
)
from mlops_cv.orchestration.handoff import GateVerdict
from mlops_cv.startup import exits_on_startup_error
from mlops_cv.tracking import champion as champion_mod
from mlops_cv.tracking import client
from mlops_cv.tracking.metric_keys import (
    LATENCY_PREFIX,
    MAP_SMALL,
    PRIMARY,
    headline_metrics,
    metric_key,
    per_class_metrics,
)
from mlops_cv.tracking.resolve import resolve_model, source_run_id

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
    p.add_argument("--split", default="test", help="dataset split to evaluate")
    p.add_argument(
        "--target",
        choices=TARGETS,
        default=SERVER,
        help="deployment target: whose champion the challenger is gated against",
    )
    p.add_argument(
        "--head",
        choices=HEADS,
        default=ONE_TO_MANY,
        help="detection head to score: the one the target exports (edge: always one-to-many)",
    )
    p.add_argument("--imgsz", type=int, default=None, help="default: the model's training imgsz")
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


def split_metrics(
    model: Any,
    data_yaml: Path,
    split: str,
    *,
    imgsz: int,
    device: str,
    batch: int,
    name: str | None = None,
    **overrides: Any,
) -> dict[str, float]:
    """``model.val`` on one split -> headline, per-class and mAP50-95_small keyed ``<split>/...``.

    ``name`` picks the val save dir. ``overrides`` go to ``model.val`` as is (``end2end=False``).
    """
    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        batch=batch,
        name=name,
        save_json=True,  # predictions.json, for mAP50-95_small
        **overrides,
    )
    metrics = headline_metrics(results.box, prefix=split)
    metrics[metric_key(split, MAP_SMALL)] = ap_small_mod.ap_small(
        ap_small_mod.load_predictions(results.save_dir),
        split_labels(data_yaml, split),
        results.names,
    )
    return metrics | per_class_metrics(
        results.maps, results.ap_class_index, results.names, prefix=metric_key(split, PRIMARY)
    )


def latency_stub(
    model: Any,
    images_dir: Path,
    *,
    imgsz: int,
    device: str,
    runs: int = 50,
    end2end: bool | None = None,
) -> dict[str, float]:
    """A quick single-configuration latency (batch 1, pre/post included) at imgsz/device.

    The rigorous multi-variant benchmark is the optimization harness's job.
    """
    return benchmark_mod.benchmark_model(
        model,
        [str(p) for p in sorted(images_dir.glob("*.jpg"))],
        imgsz=imgsz,
        device=device,
        prefix=LATENCY_PREFIX,
        warmup=3,
        iterations=runs,
        ps=(50.0, 95.0),
        end2end=end2end,
    )


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    args = build_parser(settings).parse_args(argv)

    mlflow = client.connect(settings)

    from ultralytics import YOLO

    data_yaml = dataset_yaml(settings)
    weights, model_ref = resolve_model(args.model)
    imgsz = args.imgsz or champion_mod.training_imgsz(source_run_id(args.model))
    head = exported_head(args.target, args.head)
    images_dir = settings.data.subset_dir / IMAGES_DIRNAME / args.split

    logger.info(f"evaluating {model_ref} ({head} head, imgsz {imgsz}) on {data_yaml}[{args.split}]")

    run_kwargs = {"run_id": args.run_id} if args.run_id else {"run_name": f"eval-{model_ref}"}
    with mlflow.start_run(**run_kwargs):  # type: ignore[arg-type]
        mlflow.autolog(disable=True)
        mlflow.set_tags({"eval.model": model_ref, "eval.split": args.split, "eval.head": head})
        model = YOLO(str(weights))

        candidate = split_metrics(
            model,
            data_yaml,
            args.split,
            imgsz=imgsz,
            device=args.device,
            batch=args.batch,
            end2end=head == ONE_TO_ONE,
        )
        mlflow.log_metrics(candidate)

        champion = gate_mod.fetch_champion_metrics(settings, args.target)
        thresholds = gate_mod.GateThresholds(
            min_map50_95=args.min_map50_95,
            min_map50=args.min_map50,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
            min_improvement=args.min_improvement,
        )
        gate = gate_mod.evaluate_gate(candidate, thresholds, champion, prefix=args.split)
        mlflow.set_tags(
            {gate_mod.GATE_PASSED_TAG: gate.passed, "gate.challenger_win": gate.is_challenger_win}
        )
        mlflow.log_metric("gate/passed", float(gate.passed))

        latency = None
        if args.latency_enabled:
            latency = latency_stub(
                model,
                images_dir,
                imgsz=imgsz,
                device=args.device,
                end2end=head == ONE_TO_ONE,
                runs=args.latency_runs,
            )
            mlflow.log_metrics(latency)

        work_dir = Path(tempfile.mkdtemp(prefix="eval-"))

        if args.demos_enabled and not demo_store_present(settings.data.subset_dir):
            logger.warning("subset has no demo store; skipping demo videos")
        elif args.demos_enabled:
            try:
                from mlops_cv.evaluation.visualize import load_demo_clips, render_demo_clips

                # Frames + GT come from the subset's demo store (built by the ingest pipeline).
                videos = render_demo_clips(
                    model,
                    load_demo_clips(DEMO_CLIPS),
                    settings.data.subset_dir / DEMO_DIRNAME,
                    work_dir / "demo",
                )
                for video in videos:
                    mlflow.log_artifact(str(video), artifact_path="demo")
                logger.info(f"logged {len(videos)} demo videos")
            except Exception as exc:
                logger.warning(f"demo rendering failed: {exc}")

        if args.crops_enabled:
            try:
                from mlops_cv.evaluation.error_analysis import run_error_analysis

                crops = run_error_analysis(
                    model,
                    settings.data.subset_dir / IMAGES_DIRNAME / args.split,
                    settings.data.subset_dir / LABELS_DIRNAME / args.split,
                    model.names,
                    work_dir / "error_analysis",
                )
                mlflow.log_artifacts(
                    str(work_dir / "error_analysis"), artifact_path="error_analysis"
                )
                logger.info(
                    f"logged {len(crops.low_conf_tp)} low-conf TP + "
                    f"{len(crops.high_conf_fp)} high-conf FP crops"
                )
            except Exception as exc:
                logger.warning(f"error analysis failed: {exc}")

        md, csv_path = report_mod.write_report(
            work_dir,
            candidate=candidate,
            champion=champion,
            gate=gate,
            latency=latency,
        )

        mlflow.log_artifact(str(md), artifact_path="eval")
        mlflow.log_artifact(str(csv_path), artifact_path="eval")

        logger.info(gate.summary())

    # Machine-readable verdict: orchestrators read the last stdout line (DockerOperator XCom).
    print(
        GateVerdict(
            passed=gate.passed,
            candidate_primary=gate.candidate_primary,
            champion_primary=gate.champion_primary,
        ).to_line(),
        flush=True,
    )
    return 0 if args.exit_zero or gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
