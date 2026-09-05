"""Integration smoke for the MLflow Compose stack (Postgres + RustFS + server).

Requires the stack to be up::

    docker compose -f docker-compose/docker-compose.mlflow.yml \
        --env-file docker-compose/.env.mlflow up -d --build
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid

import pytest

from mlops_cv.config import get_settings

pytestmark = pytest.mark.docker

TIMEOUT = 5


def _get(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:  # noqa: S310 (localhost only)
        return resp.status, resp.read().decode()


def _post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 (localhost only)
        return json.loads(resp.read().decode())


@pytest.fixture(scope="module")
def base_url() -> str:
    url = get_settings().mlflow.tracking_uri
    try:
        urllib.request.urlopen(f"{url}/health", timeout=TIMEOUT)  # noqa: S310
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"MLflow stack not reachable at {url}: {exc}")
    return url


def test_health_ok(base_url: str) -> None:
    status, body = _get(f"{base_url}/health")
    assert status == 200
    assert body.strip() == "OK"


def test_create_experiment(base_url: str) -> None:
    name = f"smoke-{uuid.uuid4().hex[:8]}"
    out = _post_json(f"{base_url}/api/2.0/mlflow/experiments/create", {"name": name})
    assert out.get("experiment_id"), out  # registry-capable Postgres backend answered
