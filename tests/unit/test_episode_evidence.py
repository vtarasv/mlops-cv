"""Episode evidence: the record run, its reuse, and the series it continues."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mlops_cv.monitoring.drift import WindowVerdict
from mlops_cv.monitoring.evidence import (
    DATA_VERSION_TAG,
    EPISODES_ARTIFACT_PATH,
    RECORD_RUN_NAME,
    RECORD_TAG,
    SOURCE_RUN_TAG,
    VERSION_TAG,
    EpisodeRecorder,
    open_record,
)
from mlops_cv.monitoring.monitor import DriftWindow
from mlops_cv.monitoring.resolve import ResolvedBaseline
from mlops_cv.streaming.messages import FrameRef, ModelInfo
from mlops_cv.tracking.metric_keys import DRIFT_EPISODE_METRIC
from mlops_cv.tracking.records import PARENT_TAG

TRAINING_RUN = "train-run-1"


def _resolved(version: str = "8", training_run: str = TRAINING_RUN) -> ResolvedBaseline:
    return ResolvedBaseline(
        scenes=[],
        model=ModelInfo(name="aerial-object-detector", version=version),
        data_run_id="data-run-1",
        training_run_id=training_run,
    )


def _window(*, crossed: tuple[str, ...] = ("brightness",), frames: list[FrameRef] | None = None):
    verdict = WindowVerdict(
        scores={"brightness": 3.5, "blur": 1.0},
        thresholds={"brightness": 2.6, "blur": 3.2},
        crossed=crossed,
    )
    default = [FrameRef(sequence="seqA", frame_index=i, ts_ms=1) for i in (0, 5, 10)]
    return DriftWindow(verdict=verdict, frames=tuple(frames if frames is not None else default))


@pytest.fixture
def client(registry):
    """The shared in-memory registry, holding the champion's training run."""
    registry.add_run(TRAINING_RUN)
    return registry


# --- the record run ---


def test_the_record_is_a_child_of_the_champions_training_run(client) -> None:
    """One parent holds the whole story: trained here, priced and observed beside it."""
    recorder = open_record(_resolved(), client=client)
    record = client.runs[recorder.run_id]

    assert record.data.tags[PARENT_TAG] == TRAINING_RUN
    assert record.data.tags["mlflow.runName"] == RECORD_RUN_NAME
    assert record.info.experiment_id == client.EXPERIMENT_ID  # a nested run lives in its parent's


def test_the_record_names_the_version_it_judges_not_the_alias(client) -> None:
    """The alias moves; the evidence must still identify its subject afterwards."""
    recorder = open_record(_resolved(version="8"), client=client)
    tags = client.runs[recorder.run_id].data.tags

    assert tags[RECORD_TAG] == "aerial-object-detector"
    assert tags[VERSION_TAG] == "8"
    assert tags[SOURCE_RUN_TAG] == TRAINING_RUN
    assert tags[DATA_VERSION_TAG] == "data-run-1"


def test_the_record_is_terminated_not_left_open(client) -> None:
    """SIGTERM runs no atexit handler, so a held-open run would strand RUNNING on the first stop."""
    recorder = open_record(_resolved(), client=client)
    assert client.runs[recorder.run_id].info.status == "FINISHED"


def test_a_second_monitor_reuses_the_record_rather_than_duplicating_it(client) -> None:
    first = open_record(_resolved(), client=client)
    second = open_record(_resolved(), client=client)

    assert second.run_id == first.run_id
    assert len([r for r in client.runs.values() if RECORD_TAG in r.data.tags]) == 1


def test_the_optimization_record_beside_it_is_not_mistaken_for_the_monitoring_one(
    client,
) -> None:
    """Both are children of the same parent; only the marker tag tells them apart."""
    from mlops_cv.optimize.optimize import RECORD_TAG as OPTIMIZE_TAG

    client.add_run("optimize-run", tags={PARENT_TAG: TRAINING_RUN, OPTIMIZE_TAG: "models:/x@c"})
    recorder = open_record(_resolved(), client=client)

    assert recorder.run_id != "optimize-run"
    assert client.runs[recorder.run_id].data.tags[RECORD_TAG] == "aerial-object-detector"


def test_a_promotion_reparents_the_evidence(client) -> None:
    """Resolution ran against whoever is champion now, so the restart follows the promotion."""
    client.add_run("train-run-2")
    old = open_record(_resolved(version="8"), client=client)
    new = open_record(_resolved(version="9", training_run="train-run-2"), client=client)

    assert new.run_id != old.run_id
    assert client.runs[new.run_id].data.tags[PARENT_TAG] == "train-run-2"
    assert client.runs[new.run_id].data.tags[VERSION_TAG] == "9"


# --- the episode series ---


def test_an_episode_is_exactly_one_step(client) -> None:
    recorder = open_record(_resolved(), client=client)
    recorder(_window())

    assert client.get_metric_history(recorder.run_id, "drift/brightness/score") == [
        SimpleNamespace(step=0, value=3.5)
    ]
    assert client.get_metric_history(recorder.run_id, "drift/brightness/threshold") == [
        SimpleNamespace(step=0, value=2.6)
    ]
    assert len(client.get_metric_history(recorder.run_id, DRIFT_EPISODE_METRIC)) == 1


def test_episodes_advance_the_step_within_one_monitor(client) -> None:
    recorder = open_record(_resolved(), client=client)
    recorder(_window())
    recorder(_window())

    steps = [m.step for m in client.get_metric_history(recorder.run_id, DRIFT_EPISODE_METRIC)]
    assert steps == [0, 1]


def test_a_restarted_monitor_continues_the_series(client) -> None:
    """The series belongs to the model version, not to the process that happened to observe it."""
    first = open_record(_resolved(), client=client)
    first(_window())
    first(_window())

    second = open_record(_resolved(), client=client)
    assert second.step == 2
    second(_window())

    steps = [m.step for m in client.get_metric_history(second.run_id, DRIFT_EPISODE_METRIC)]
    assert steps == [0, 1, 2]


def test_a_fresh_record_starts_at_step_zero(client) -> None:
    assert open_record(_resolved(), client=client).step == 0


# --- the footage behind an episode ---


def test_the_frames_are_recorded_by_address(client) -> None:
    recorder = open_record(_resolved(), client=client)
    recorder(_window())

    run_id, text, artifact_file = client.texts[0]
    payload = json.loads(text)
    assert run_id == recorder.run_id
    assert artifact_file == f"{EPISODES_ARTIFACT_PATH}/episode-0000.json"
    assert payload["episode"] == 0
    assert payload["where"] == "seqA/0-10"
    assert payload["crossed"] == ["brightness"]
    assert payload["scores"]["brightness"] == 3.5
    assert payload["thresholds"]["brightness"] == 2.6
    assert payload["ranges"] == [{"sequence": "seqA", "first": 0, "last": 10, "n_frames": 3}]


def test_each_episode_gets_its_own_artifact(client) -> None:
    recorder = open_record(_resolved(), client=client)
    recorder(_window())
    recorder(_window())

    assert [f for _, _, f in client.texts] == [
        f"{EPISODES_ARTIFACT_PATH}/episode-0000.json",
        f"{EPISODES_ARTIFACT_PATH}/episode-0001.json",
    ]


def test_a_failed_write_costs_its_own_step_not_the_next_one(client) -> None:
    """Reusing a step would put two points on one index and misfile the next episode's footage."""
    recorder = open_record(_resolved(), client=client)

    def refuse(run_id: str, text: str, artifact_file: str) -> None:
        raise RuntimeError("artifact store unreachable")

    client.log_text = refuse  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        recorder(_window())

    client.log_text = type(client).log_text.__get__(client)  # type: ignore[method-assign]
    recorder(_window())

    steps = [m.step for m in client.get_metric_history(recorder.run_id, DRIFT_EPISODE_METRIC)]
    assert steps == [0, 1]  # the failed episode kept its own step
    assert [f for _, _, f in client.texts] == [f"{EPISODES_ARTIFACT_PATH}/episode-0001.json"]


def test_the_recorder_needs_no_open_run(client) -> None:
    """Everything is written by run id, so nothing is left active between episodes."""
    recorder = EpisodeRecorder(client, "record-x", step=4)
    client.add_run("record-x")
    recorder(_window(crossed=("brightness", "contrast")))

    assert client.get_metric_history("record-x", DRIFT_EPISODE_METRIC)[0].value == 2.0
    assert client.texts[0][2] == f"{EPISODES_ARTIFACT_PATH}/episode-0004.json"
