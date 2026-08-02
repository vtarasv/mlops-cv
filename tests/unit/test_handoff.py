"""Unit tests for the Container contract (DAG <-> task-container handoff).

The point of these tests: hold the two ends of the seam together in plain CI —
payload emit/parse round-trips, and each command builder's argv fed to the *real*
parser of the entrypoint it targets.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlops_cv.config import load_settings
from mlops_cv.orchestration.handoff import (
    BEST_WEIGHTS_RELPATH,
    GateVerdict,
    TrainHandoff,
    best_weights_uri,
    evaluate_cmd,
    ingest_cmd,
    optimize_cmd,
    profile_cmd,
    train_cmd,
)


def _settings():
    return load_settings(base_dir="/nonexistent")  # model defaults, no env files


# --- payload round-trips ---------------------------------------------------------------


def test_train_handoff_round_trip() -> None:
    handoff = TrainHandoff(version="3", run_id="abc123")
    assert TrainHandoff.parse(handoff.to_line()) == handoff


def test_gate_verdict_round_trip() -> None:
    verdict = GateVerdict(passed=True, candidate_primary=0.216, champion_primary=None)
    assert GateVerdict.parse(verdict.to_line()) == verdict


def test_wire_format_is_stable() -> None:
    """The exact bytes on the wire — an older parser must recognize a newer emitter."""
    assert TrainHandoff(version="3", run_id="abc123").to_line() == (
        '{"version": "3", "run_id": "abc123"}'
    )
    assert GateVerdict(passed=False, candidate_primary=0.2, champion_primary=0.21).to_line() == (
        '{"passed": false, "candidate_primary": 0.2, "champion_primary": 0.21}'
    )


def test_parse_tolerates_extra_keys() -> None:
    # The airflow and train images build separately; a newer emitter adding a key
    # must not break an older parser.
    parsed = TrainHandoff.parse('{"version": "3", "run_id": "abc", "brand_new": 1}')
    assert parsed == TrainHandoff(version="3", run_id="abc")


def test_parse_coerces_numeric_version() -> None:
    # Parity with the DAG's previous str(info["version"]).
    assert TrainHandoff.parse('{"version": 3, "run_id": "abc"}').version == "3"


def test_parse_fails_loudly_on_garbage() -> None:
    with pytest.raises(ValidationError):
        TrainHandoff.parse("Results saved to runs/detect/train")  # a stray last stdout line
    with pytest.raises(ValidationError):
        GateVerdict.parse('{"candidate_primary": 0.2}')  # missing "passed"


def test_train_handoff_keys_match_dag_templates() -> None:
    # The DAG's Jinja templates index the XCom dict by these keys
    # ({{ ti.xcom_pull(...)['run_id'] }}); promote reads 'version'.
    assert set(TrainHandoff.model_fields) == {"version", "run_id"}


# --- weights location ------------------------------------------------------------------


def test_best_weights_uri() -> None:
    assert best_weights_uri("abc123") == "runs:/abc123/weights/best.pt"
    assert best_weights_uri("abc123").endswith(BEST_WEIGHTS_RELPATH)


# --- command builders vs the real parsers ----------------------------------------------


def test_train_cmd_parses_with_real_parser() -> None:
    from mlops_cv.training.train import build_parser

    cmd = train_cmd(epochs=7)
    assert cmd[:3] == ["python", "-m", "mlops_cv.training.train"]
    args = build_parser(_settings()).parse_args(cmd[3:])
    assert args.epochs == 7


def test_train_cmd_passes_jinja_through() -> None:
    # The DAG hands Airflow templates as values; the builder must not mangle them.
    assert "{{ params.epochs }}" in train_cmd(epochs="{{ params.epochs }}")


def test_evaluate_cmd_parses_with_real_parser() -> None:
    from mlops_cv.eval.evaluate import build_parser

    cmd = evaluate_cmd(run_id="abc123")
    assert cmd[:3] == ["python", "-m", "mlops_cv.eval.evaluate"]
    args = build_parser(_settings()).parse_args(cmd[3:])
    assert args.model == best_weights_uri("abc123")
    assert args.run_id == "abc123"
    assert args.exit_zero is True
    assert args.batch == 8


def test_ingest_cmd_parses_with_real_options() -> None:
    pytest.importorskip("apache_beam")
    from apache_beam.options.pipeline_options import StandardOptions

    from mlops_cv.pipelines.ingest_pipeline import IngestOptions

    cmd = ingest_cmd(raw_dir="/data/raw", output_dir="/data/subset")
    assert cmd[:3] == ["python", "-m", "mlops_cv.pipelines.ingest_pipeline"]
    options = IngestOptions(cmd[3:])
    assert options.raw_dir == "/data/raw"
    assert options.output_dir == "/data/subset"
    assert options.view_as(StandardOptions).runner == "DirectRunner"


def test_profile_cmd_parses_with_real_options() -> None:
    pytest.importorskip("apache_beam")
    from apache_beam.options.pipeline_options import StandardOptions

    from mlops_cv.pipelines.profile_pipeline import ProfileOptions

    cmd = profile_cmd(input_dir="/data/subset")
    assert cmd[:3] == ["python", "-m", "mlops_cv.pipelines.profile_pipeline"]
    options = ProfileOptions(cmd[3:])
    assert options.input_dir == "/data/subset"
    assert options.view_as(StandardOptions).runner == "DirectRunner"


def test_optimize_cmd_parses_with_real_parser() -> None:
    """optimize shares evaluate's addressing: one run_id keys the whole container chain."""
    from mlops_cv.optimize.optimize import build_parser, model_uri

    cmd = optimize_cmd(run_id="abc123")
    assert cmd[:3] == ["python", "-m", "mlops_cv.optimize"]
    settings = _settings()
    args = build_parser(settings).parse_args(cmd[3:])
    assert args.model == best_weights_uri("abc123")  # never the alias — no promotion race
    assert args.run_id == "abc123"
    assert model_uri(args, settings) == best_weights_uri("abc123")


def test_optimize_cmd_renders_an_airflow_template() -> None:
    template = "{{ ti.xcom_pull(task_ids='parse_train_output')['run_id'] }}"
    cmd = optimize_cmd(run_id=template)
    assert best_weights_uri(template) in cmd
    assert template in cmd


def test_optimize_defaults_to_the_champion_alias_without_addressing() -> None:
    from mlops_cv.optimize.optimize import build_parser, model_uri

    settings = _settings()
    args = build_parser(settings).parse_args([])
    expected = f"models:/{settings.mlflow.registered_model}@{settings.mlflow.champion_alias}"
    assert model_uri(args, settings) == expected
