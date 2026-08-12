"""Integration smoke for the serving Compose stack (the containerized detection service).

Requires the stack's GPU tenant to be up::

    make serving-up
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

from mlops_cv.config import get_settings

pytestmark = pytest.mark.docker

TIMEOUT = 5


def _get(url: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:  # noqa: S310 (localhost only)
        return resp.status, resp.read().decode(), resp.headers.get_content_type()


@pytest.fixture(scope="module")
def base_url() -> str:
    url = f"http://localhost:{os.environ.get('SERVING_HOST_PORT', '8000')}"
    try:
        urllib.request.urlopen(f"{url}/health", timeout=TIMEOUT)  # noqa: S310
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"serving stack not reachable at {url}: {exc}")
    return url


def test_health_reports_the_model_it_resolved_in_network(base_url: str) -> None:
    """A healthy container means the in-network champion lookup and the graph pull worked."""
    status, body, _ = _get(f"{base_url}/health")
    assert status == 200
    health = json.loads(body)
    assert health["status"] == "ok"
    assert health["model"]["name"] == get_settings().mlflow.registered_model
    assert health["model"]["version"], health  # a resolved registry version, not a blank
    assert health["imgsz"] == get_settings().optimize.server_imgsz


def test_metrics_are_scrapeable_through_the_published_port(base_url: str) -> None:
    status, body, content_type = _get(f"{base_url}/metrics")
    assert status == 200
    assert content_type == "text/plain"
    assert "http_requests_total" in body
