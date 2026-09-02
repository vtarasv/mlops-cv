"""Integration smoke for the observability half of the serving stack.

Requires the stack to be up::

    make serving-up
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.docker

TIMEOUT = 5
SCRAPE_WAIT_S = 30  # a fresh stack needs a scrape cycle or two before targets report up


def _get_json(url: str) -> dict | list:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:  # noqa: S310 (localhost only)
        return json.loads(resp.read().decode())


def _base_url_or_skip(service: str, port_env: str, default_port: str, probe: str) -> str:
    url = f"http://localhost:{os.environ.get(port_env, default_port)}"
    try:
        urllib.request.urlopen(f"{url}{probe}", timeout=TIMEOUT)  # noqa: S310
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"{service} not reachable at {url}: {exc}")
    return url


@pytest.fixture(scope="module")
def prometheus_url() -> str:
    return _base_url_or_skip("prometheus", "PROMETHEUS_HOST_PORT", "9090", "/-/ready")


@pytest.fixture(scope="module")
def grafana_url() -> str:
    return _base_url_or_skip("grafana", "GRAFANA_HOST_PORT", "3000", "/api/health")


def test_prometheus_sees_exactly_four_targets_and_all_are_up(prometheus_url: str) -> None:
    """serving + drift + redpanda + dcgm scraped and healthy, and nothing else configured."""
    deadline = time.monotonic() + SCRAPE_WAIT_S
    while True:
        targets = _get_json(f"{prometheus_url}/api/v1/targets")["data"]["activeTargets"]  # type: ignore
        health = {t["labels"]["job"]: t["health"] for t in targets}
        if set(health) == {"serving", "drift", "redpanda", "dcgm"} and set(health.values()) == {
            "up"
        }:
            break
        if time.monotonic() > deadline:
            pytest.fail(f"targets never converged to four up: {health}")
        time.sleep(2)


def test_grafana_serves_the_four_provisioned_dashboards_anonymously(grafana_url: str) -> None:
    """Zero clicks from `up` to dashboards: anonymous access + file provisioning both work."""
    dashboards = _get_json(f"{grafana_url}/api/search?type=dash-db")
    assert {d["uid"] for d in dashboards} == {"serving", "gpu", "streaming", "drift"}
