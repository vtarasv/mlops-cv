"""Shared doubles: an in-memory MLflow registry, an onnxruntime-shaped session, PNG uploads.

The tracking tests (champion resolution, record runs, episode evidence) all drive the same
:class:`FakeRegistry`; the runtime tests drive the detector directly and the app tests drive
it through HTTP, both on the same fake session.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from types import SimpleNamespace

import numpy as np
import pytest

# --- the MLflow registry -------------------------------------------------------------------


def _missing(message: str) -> Exception:
    """What the real client raises for an unknown alias/version/run."""
    from mlflow.exceptions import MlflowException

    return MlflowException(message)


class FakeRun:
    """An mlflow ``Run`` as the tracking modules read it: ``info`` + ``data``."""

    def __init__(
        self,
        run_id: str,
        experiment_id: str,
        tags: Mapping[str, str] | None = None,
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        self.info = SimpleNamespace(run_id=run_id, experiment_id=experiment_id, status="RUNNING")
        self.data = SimpleNamespace(tags=dict(tags or {}), metrics=dict(metrics or {}), params={})


class FakeVersion:
    """A registered model version as the registry returns it."""

    def __init__(
        self,
        version: str = "7",
        run_id: str | None = "train-run",
        tags: Mapping[str, str] | None = None,
    ) -> None:
        self.version = version
        self.run_id = run_id
        self.tags = dict(tags or {})
        self.source = f"runs:/{run_id}/weights/best.pt"


class FakeRegistry:
    """An in-memory ``MlflowClient``: the slice the tracking modules use, recording every write.

    Build state with :meth:`promote`, :meth:`add_version` and :meth:`add_run`; read the queries
    back from ``asked``.
    """

    EXPERIMENT_ID = "7"

    def __init__(self) -> None:
        self.runs: dict[str, FakeRun] = {}
        self.versions: dict[str, FakeVersion] = {}
        self.aliases: dict[str, str] = {}
        self.history: dict[tuple[str, str], list[SimpleNamespace]] = {}
        self.texts: list[tuple[str, str, str]] = []
        self.asked: list[tuple] = []
        self._counter = 0

    # -- building state ------------------------------------------------------------------
    def add_run(
        self,
        run_id: str,
        tags: Mapping[str, str] | None = None,
        metrics: Mapping[str, float] | None = None,
    ) -> FakeRun:
        self.runs[run_id] = FakeRun(run_id, self.EXPERIMENT_ID, tags, metrics)
        return self.runs[run_id]

    def add_version(
        self,
        version: str = "7",
        run_id: str | None = "train-run",
        tags: Mapping[str, str] | None = None,
    ) -> FakeVersion:
        """Register a version; its training run is created unless it already exists."""
        entity = FakeVersion(version, run_id, tags)
        self.versions[version] = entity
        if run_id is not None and run_id not in self.runs:
            self.add_run(run_id)
        return entity

    def promote(
        self,
        version: str = "7",
        run_id: str | None = "train-run",
        tags: Mapping[str, str] | None = None,
        *,
        alias: str = "champion",
    ) -> FakeVersion:
        """Register a version and point ``alias`` at it."""
        entity = self.add_version(version, run_id, tags)
        self.aliases[alias] = version
        return entity

    # -- the model registry ---------------------------------------------------------------
    def get_model_version_by_alias(self, name: str, alias: str) -> FakeVersion:
        self.asked.append(("alias", name, alias))
        if alias not in self.aliases:
            raise _missing(f"Registered model alias {alias} not found.")
        return self.versions[self.aliases[alias]]

    def get_model_version(self, name: str, version: str) -> FakeVersion:
        self.asked.append(("version", name, version))
        if version not in self.versions:
            raise _missing(f"Model Version (name={name}, version={version}) not found")
        return self.versions[version]

    def search_model_versions(self, filter_string: str) -> list[FakeVersion]:
        self.asked.append(("search", filter_string))
        match = re.fullmatch(r"run_id='(.*)' and name='(.*)'", filter_string)
        assert match, f"unsupported filter {filter_string!r}"
        run_id = match.group(1)
        return [v for v in self.versions.values() if v.run_id == run_id]

    # -- runs ---------------------------------------------------------------------------------
    def get_run(self, run_id: str) -> FakeRun:
        if run_id not in self.runs:
            raise _missing(f"Run '{run_id}' not found")
        return self.runs[run_id]

    def search_runs(self, experiment_ids: Iterable[str], filter_string: str = "") -> list[FakeRun]:
        match = re.fullmatch(r"tags\.(\S+) = '(.*)'", filter_string)
        assert match, f"unsupported filter {filter_string!r}"
        key, value = match.groups()
        wanted = set(experiment_ids)
        return [
            run
            for run in self.runs.values()
            if run.info.experiment_id in wanted and run.data.tags.get(key) == value
        ]

    def create_run(self, experiment_id: str, tags: Mapping[str, str], run_name: str) -> FakeRun:
        self._counter += 1
        run_id = f"record-{self._counter}"
        self.runs[run_id] = FakeRun(run_id, experiment_id, {**tags, "mlflow.runName": run_name})
        return self.runs[run_id]

    def set_terminated(self, run_id: str, status: str) -> None:
        self.runs[run_id].info.status = status

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        self.history.setdefault((run_id, key), []).append(SimpleNamespace(step=step, value=value))
        self.runs[run_id].data.metrics[key] = value

    def get_metric_history(self, run_id: str, key: str) -> list[SimpleNamespace]:
        return self.history.get((run_id, key), [])

    def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        self.texts.append((run_id, text, artifact_file))


@pytest.fixture
def registry() -> FakeRegistry:
    """An empty in-memory registry — promote a version into it, or leave it empty."""
    return FakeRegistry()


# --- the onnxruntime session -----------------------------------------------------------------

# The merged VisDrone class map, as the exporter embeds it in the graph metadata.
GRAPH_NAMES = {0: "person", 1: "vehicle", 2: "two-three-wheeler"}
GRAPH_METADATA = {
    "names": str(GRAPH_NAMES),
    "imgsz": "[640, 640]",
    "task": "detect",
    "batch": "1",
}


class FakeSession:
    """An onnxruntime session as the detector uses it: metadata, one input, one output."""

    def __init__(self, output: np.ndarray, metadata: dict[str, str] | None = None) -> None:
        self.output = output
        self.metadata = GRAPH_METADATA if metadata is None else metadata
        self.feeds: list[dict[str, np.ndarray]] = []

    def get_modelmeta(self) -> SimpleNamespace:
        return SimpleNamespace(custom_metadata_map=dict(self.metadata))

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="images", shape=[1, 3, 640, 640])]

    def get_providers(self) -> list[str]:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def run(self, output_names, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(feed)
        return [self.output]


@pytest.fixture
def fake_session() -> type[FakeSession]:
    """The session double's class — tests construct it with the output they want back."""
    return FakeSession


@pytest.fixture
def graph_names() -> dict[int, str]:
    """The class map the fake graph advertises."""
    return dict(GRAPH_NAMES)


@pytest.fixture
def graph_metadata() -> dict[str, str]:
    """The fake graph's embedded metadata, for tests that corrupt one field of it."""
    return dict(GRAPH_METADATA)


@pytest.fixture
def png_upload() -> Callable[[int, int], bytes]:
    """``png_upload(height, width)`` -> a lossless gradient PNG, decoding to exactly itself."""

    def encode(height: int, width: int) -> bytes:
        import cv2

        rows = np.linspace(0, 255, height, dtype=np.uint8)[:, None, None]
        cols = np.linspace(0, 255, width, dtype=np.uint8)[None, :, None]
        image = np.broadcast_to(rows // 2 + cols // 2, (height, width, 3)).astype(np.uint8)
        ok, buffer = cv2.imencode(".png", image)
        assert ok
        return buffer.tobytes()

    return encode
