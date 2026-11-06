"""Unit tests for the tuning-stage runner's config and argv (no GPU, no mlflow)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from mlops_cv.config import load_settings
from mlops_cv.training.train import build_parser
from mlops_cv.tuning.common import load_stage
from mlops_cv.tuning.yolo26_visdrone_vid.tune import trial_argv, trial_env

ROOT = Path(__file__).parents[2]
STAGES = ROOT / "configs" / "tuning" / "yolo26-visdrone-vid"

STAGE = """
defaults: {patience: 0, close_mosaic: 10, imgsz: 640}
trials:
  - {name: a, model: yolo26n}
  - {name: b, model: yolo26m, imgsz: 1280, scale: [0.5, 3.0], cache: ram}
"""


def _stage(tmp_path: Path, text: str = STAGE) -> Path:
    path = tmp_path / "stage1.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_knobs_reach_model_train_with_their_types(tmp_path: Path) -> None:
    """Flags stay flags; the rest round-trips through ``--set`` as YAML (a list stays a list)."""
    argv = trial_argv("stage1-b", load_stage(_stage(tmp_path))["stage1-b"])
    assert argv[1:3] == ["-m", "mlops_cv.training"]
    args = build_parser(load_settings(base_dir="/nonexistent")).parse_args(argv[3:])
    assert (args.model, args.imgsz, args.register, args.run_name) == (
        "yolo26m",
        1280,
        False,
        "stage1-b",
    )
    assert dict(args.overrides) == {
        "patience": 0,
        "close_mosaic": 10,
        "scale": [0.5, 3.0],
        "cache": "ram",
    }


def test_a_dataset_knob_reaches_training_as_its_dataset_yaml() -> None:
    knobs = {"model": "yolo26n", "dataset": "configs/x.yaml"}
    assert "dataset" not in " ".join(trial_argv("stage1-a", knobs))
    assert trial_env(knobs)["DATA__DATASET_YAML"] == "configs/x.yaml"
    assert trial_env({"model": "yolo26n"}).get("DATA__DATASET_YAML") == os.environ.get(
        "DATA__DATASET_YAML"
    )


def test_the_autolabel_stage_is_the_final_s_on_the_autolabel_split() -> None:
    from mlops_cv.tuning.autolabel_visdrone_vid.visdrone_vid import AUTOLABEL_SPLIT

    final = load_stage(STAGES / "stage2.yaml")["stage2-s1280"]
    (trial,) = load_stage(STAGES / "autolabels.yaml").values()
    dataset = ROOT / trial.pop("dataset")
    assert trial == final
    base = yaml.safe_load((ROOT / "configs/datasets/VisDrone-VID-merged.yaml").read_text())
    assert yaml.safe_load(dataset.read_text()) == base | {"train": f"images/{AUTOLABEL_SPLIT}"}
