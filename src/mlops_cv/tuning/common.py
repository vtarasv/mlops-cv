"""Helpers both tuning cases share: stage files, finished trials, and scoring like a trial."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TUNE_EXPERIMENT = "aerial-object-detection-tune"

# val's own floor and box cap (``model.val`` defaults): detections scored here are predicted with
# them. For a class-mapped model, the cap applies before the mapping drops classes, as in the model.
CONF = 0.001
MAX_DET = 300


def load_stage(path: Path) -> dict[str, dict[str, Any]]:
    """``{run name: knobs}``, each trial's knobs over the stage's ``defaults``.

    A trial's section (a mapping) updates the default's keys, one level deep; any value inside
    one replaces the default's whole.
    """
    stage = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = stage.get("defaults", {})
    trials: dict[str, dict[str, Any]] = {}
    for entry in stage["trials"]:
        knobs = {**defaults}
        for key, value in entry.items():
            base = knobs.get(key)
            knobs[key] = {**base, **value} if isinstance(base, dict) else value
        name = f"{path.stem}-{knobs.pop('name')}"
        if name in trials:
            raise ValueError(f"trial {name} repeats in {path}")
        trials[name] = knobs
    return trials


def finished_trial(mlflow: Any, name: str) -> Any:
    """The latest finished run named ``name`` in the tune experiment."""
    runs = mlflow.search_runs(
        experiment_names=[TUNE_EXPERIMENT],
        filter_string=f"attributes.run_name = '{name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise ValueError(f"no finished trial run named {name}")
    return runs[0]


def readings(
    detections: Mapping[str, np.ndarray],
    data_yaml: Path,
    split: str,
    prefix: str,
    frames: Collection[str] | None = None,
) -> dict[str, float]:
    """Merged detections scored on ``split`` (or its ``frames``) -> trial keys under ``prefix``."""
    from mlops_cv.evaluation.score_detections import score_with_ap_small
    from mlops_cv.tracking.metric_keys import (
        MAP_SMALL,
        PRIMARY,
        headline_metrics,
        metric_key,
        per_class_metrics,
    )

    metrics, ap_s = score_with_ap_small(detections, data_yaml, split, frames)
    return (
        headline_metrics(metrics.box, prefix=prefix)
        | {metric_key(prefix, MAP_SMALL): ap_s}
        | per_class_metrics(
            metrics.maps,
            metrics.ap_class_index,
            metrics.names,
            prefix=metric_key(prefix, PRIMARY),
        )
    )
