"""Where a drift episode is kept after the graph scrolls away.

A Prometheus counter says *that* an episode happened and forgets it at the end of its retention;
the frames it covered are nowhere. This module records each episode as one **step** on a run that
belongs to the model being judged — a child of the champion's training run.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from mlops_cv.tracking.metric_keys import DRIFT_EPISODE_METRIC, episode_metrics

if TYPE_CHECKING:
    from mlops_cv.monitoring.monitor import DriftWindow
    from mlops_cv.monitoring.resolve import ResolvedBaseline

logger = logging.getLogger(__name__)

# MLflow nesting is this tag.
PARENT_TAG = "mlflow.parentRunId"

RECORD_TAG = "monitor.model"
VERSION_TAG = "monitor.version"
SOURCE_RUN_TAG = "monitor.source_run"
DATA_VERSION_TAG = "monitor.data_version"

RECORD_RUN_NAME = "monitor"

EPISODES_ARTIFACT_PATH = "episodes"


def record_tags(resolved: ResolvedBaseline) -> dict[str, str]:
    """What this evidence is about, in full."""
    return {
        RECORD_TAG: resolved.model.name,
        VERSION_TAG: resolved.model.version,
        SOURCE_RUN_TAG: resolved.training_run_id,
        DATA_VERSION_TAG: resolved.data_run_id,
    }


def episode_payload(window: DriftWindow, step: int) -> dict[str, Any]:
    """One episode as it is stored: the reading that opened it, and the footage behind it."""
    verdict = window.verdict
    return {
        "episode": step,
        "where": window.where(),
        "crossed": list(verdict.crossed),
        "scores": dict(verdict.scores),
        "thresholds": dict(verdict.thresholds),
        "n_frames": len(window.frames),
        "ranges": window.spans(),
    }


def existing_record(client: Any, parent_run_id: str, experiment_id: str) -> str | None:
    """This model version's monitoring record, if some earlier monitor already opened it."""
    children = client.search_runs(
        [experiment_id], filter_string=f"tags.{PARENT_TAG} = '{parent_run_id}'"
    )
    return next((r.info.run_id for r in children if RECORD_TAG in r.data.tags), None)


def next_step(client: Any, run_id: str) -> int:
    """Where this monitor's episodes continue the series a previous one left."""
    history = client.get_metric_history(run_id, DRIFT_EPISODE_METRIC)
    return max((m.step for m in history), default=-1) + 1


class EpisodeRecorder:
    """The monitor's episode sink: one episode in, one step on the record run out."""

    def __init__(self, client: Any, run_id: str, *, step: int = 0) -> None:
        self.client = client
        self.run_id = run_id
        self.step = step

    def __call__(self, window: DriftWindow) -> None:
        """Record one episode: its readings as metrics, its footage as a small artifact."""
        step, self.step = self.step, self.step + 1
        for key, value in episode_metrics(window.verdict).items():
            self.client.log_metric(self.run_id, key, value, step=step)
        self.client.log_text(
            self.run_id,
            json.dumps(episode_payload(window, step), indent=2),
            f"{EPISODES_ARTIFACT_PATH}/episode-{step:04d}.json",
        )
        logger.info(f"episode {step} recorded on run {self.run_id}")


def open_record(resolved: ResolvedBaseline, *, client: Any | None = None) -> EpisodeRecorder:
    """Find or open the monitoring record for the resolved champion, ready at its next step.

    A promotion re-parents the evidence for free: resolution ran against whichever version is
    champion *now*, so a restarted monitor opens the record under that version's training run
    without knowing a promotion happened.
    """
    if client is None:
        from mlflow import MlflowClient

        client = MlflowClient()

    parent = resolved.training_run_id
    experiment_id = client.get_run(parent).info.experiment_id
    run_id = existing_record(client, parent, experiment_id)
    if run_id is None:
        run = client.create_run(
            experiment_id,
            tags={PARENT_TAG: parent, **record_tags(resolved)},
            run_name=RECORD_RUN_NAME,
        )
        run_id = run.info.run_id
        # Terminated on creation: the ledger outlives every monitor that appends to it.
        client.set_terminated(run_id, "FINISHED")
        logger.info(f"opened the monitoring record {run_id} under training run {parent}")
    step = next_step(client, run_id)
    logger.info(
        f"recording model version {resolved.model.version}'s drift episodes on run {run_id}, "
        f"continuing from step {step}"
    )
    return EpisodeRecorder(client, run_id, step=step)
