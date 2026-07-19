"""Integration smoke for the Airflow CT Compose stack (Postgres + api-server + scheduler +
dag-processor + triggerer).

Requires the stack to be up::

    make airflow-up
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.docker

TIMEOUT = 5
HEALTH = "/api/v2/monitor/health"  # unauthenticated monitoring endpoint (Airflow 3)
# LocalExecutor worker components the health endpoint reports, besides the metadatabase.
COMPONENTS = ("scheduler", "dag_processor", "triggerer")


def _get(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:  # noqa: S310 (localhost only)
        return resp.status, resp.read().decode()


@pytest.fixture(scope="module")
def base_url() -> str:
    url = f"http://localhost:{os.environ.get('AIRFLOW_PORT', '8080')}"
    try:
        urllib.request.urlopen(f"{url}{HEALTH}", timeout=TIMEOUT)  # noqa: S310
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"Airflow stack not reachable at {url}: {exc}")
    return url


def test_health_ok(base_url: str) -> None:
    status, body = _get(f"{base_url}{HEALTH}")
    assert status == 200
    health = json.loads(body)
    assert health["metadatabase"]["status"] == "healthy", health  # Postgres backend answered


def test_components_healthy(base_url: str) -> None:
    _, body = _get(f"{base_url}{HEALTH}")
    health = json.loads(body)
    unhealthy = {
        c: health.get(c) for c in COMPONENTS if health.get(c, {}).get("status") != "healthy"
    }
    assert not unhealthy, unhealthy
