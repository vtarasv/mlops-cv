"""The Container contract: what crosses the CT DAG <-> task-container seam.

The DAG supplies each container's command line (built here); a container with something
to say replies with a single machine-readable JSON line as its last stdout line
(DockerOperator's XCom), emitted and parsed here. Both ends import this module.

Import rules: the Airflow scheduler imports this at DAG-parse time, so it must stay
pydantic + stdlib only — no mlflow/ultralytics/beam imports. Command-builder parameters
accept plain values or Airflow Jinja template strings (the DAG passes e.g.
``"{{ params.epochs }}"`` for Airflow to render); payload parsing tolerates unknown JSON
keys because the airflow and task images are built separately and may skew.
"""

from __future__ import annotations

import json
from typing import Self

from pydantic import BaseModel, ConfigDict

# Where ultralytics leaves the val-selected checkpoint, relative to a run's artifact root.
BEST_WEIGHTS_RELPATH = "weights/best.pt"


def best_weights_uri(run_id: str) -> str:
    """MLflow ``runs:/`` URI of a training run's best checkpoint."""
    return f"runs:/{run_id}/{BEST_WEIGHTS_RELPATH}"


class _HandoffPayload(BaseModel):
    """One JSON line crossing the Container contract (extra keys ignored: image skew)."""

    # extra="ignore" is pydantic's default — declared so the skew tolerance is contract, not
    # accident. Numeric-looking str fields coerce (parity with the DAG's previous str(...) parse).
    model_config = ConfigDict(extra="ignore", coerce_numbers_to_str=True)

    def to_line(self) -> str:
        """The wire form: print this as the container's last stdout line."""
        return json.dumps(self.model_dump())

    @classmethod
    def parse(cls, line: str) -> Self:
        """Parse a container's reply line; raises ``pydantic.ValidationError`` loudly."""
        return cls.model_validate_json(line)


class TrainHandoff(_HandoffPayload):
    """Train's reply: the registered model version produced and the run it lives on."""

    version: str
    run_id: str


class GateVerdict(_HandoffPayload):
    """Evaluate's reply: the champion/challenger decision the promotion branch reads.

    ``champion_primary`` is ``None`` on a bootstrap win (no champion alias yet).
    """

    passed: bool
    candidate_primary: float
    champion_primary: float | None = None


def train_cmd(epochs: int | str) -> list[str]:
    """Train-container argv; the container replies with a ``TrainHandoff`` line."""
    return ["python", "-m", "mlops_cv.training", "--epochs", str(epochs)]


def evaluate_cmd(run_id: str, batch: int | str = 8) -> list[str]:
    """Evaluate-container argv; the container replies with a ``GateVerdict`` line.

    Standing policy lives here, not in the DAG: the model under test is the training
    run's best checkpoint, test metrics land on that same run (``--run-id``), and a
    challenger loss is a verdict for the branch, not a task failure (``--exit-zero``).
    """
    return [
        "python",
        "-m",
        "mlops_cv.evaluation",
        "--model",
        best_weights_uri(run_id),
        "--run-id",
        run_id,
        "--exit-zero",
        "--batch",
        str(batch),
    ]


def optimize_cmd(run_id: str) -> list[str]:
    """Optimize-container argv: build + benchmark a promoted model's serving variants. No reply."""
    return [
        "python",
        "-m",
        "mlops_cv.optimize",
        "--model",
        best_weights_uri(run_id),
        "--run-id",
        run_id,
    ]


def ingest_cmd(raw_dir: str, output_dir: str) -> list[str]:
    """Ingest-container argv (Beam DirectRunner): raw -> subset + demo store. No reply."""
    return [
        "python",
        "-m",
        "mlops_cv.pipelines.ingest_pipeline",
        "--runner",
        "DirectRunner",
        f"--raw-dir={raw_dir}",
        f"--output-dir={output_dir}",
    ]


def profile_cmd(input_dir: str) -> list[str]:
    """Profile-container argv (Beam DirectRunner): quality report + drift baseline. No reply."""
    return [
        "python",
        "-m",
        "mlops_cv.pipelines.profile_pipeline",
        "--runner",
        "DirectRunner",
        f"--input-dir={input_dir}",
    ]
