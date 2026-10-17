"""Values the compose env files and the Python defaults both carry must agree."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import pytest

from mlops_cv.config.settings import (
    MlflowSettings,
    MonitoringSettings,
    ServingSettings,
    StreamingSettings,
)
from mlops_cv.streaming.messages import FRAME_MAX_MESSAGE_BYTES

COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose"
# The nested models are plain BaseModels: their defaults never read .env or the OS env.


def _env(stack: str) -> dict[str, str]:
    text = (COMPOSE / f".env.{stack}").read_text(encoding="utf-8")
    return dict(
        line.split("=", 1) for line in text.splitlines() if line and not line.startswith("#")
    )


@pytest.mark.parametrize("stack", ["mlflow", "airflow", "serving"])
def test_every_stack_dials_mlflow_on_the_port_host_clients_use(stack: str) -> None:
    assert int(_env(stack)["MLFLOW_PORT"]) == urlsplit(MlflowSettings().tracking_uri).port


def test_broker_publishes_on_the_port_host_clients_dial() -> None:
    port = StreamingSettings().bootstrap_servers.rsplit(":", 1)[1]
    assert _env("streaming")["KAFKA_HOST_PORT"] == port


def test_the_broker_creates_the_topics_the_clients_use() -> None:
    env, streaming = _env("streaming"), StreamingSettings()
    assert (env["RAW_FRAMES_TOPIC"], env["DETECTIONS_TOPIC"], env["ALERTS_TOPIC"]) == (
        streaming.raw_frames_topic,
        streaming.detections_topic,
        streaming.alerts_topic,
    )


def test_the_broker_frame_cap_matches_the_client_cap() -> None:
    assert int(_env("streaming")["FRAME_MAX_MESSAGE_BYTES"]) == FRAME_MAX_MESSAGE_BYTES


def test_service_and_monitor_listen_where_the_serving_stack_says() -> None:
    env = _env("serving")
    assert int(env["SERVING_PORT"]) == ServingSettings().port
    assert int(env["MONITOR_METRICS_PORT"]) == MonitoringSettings().metrics_port
