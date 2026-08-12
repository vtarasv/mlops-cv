"""The observability config is code — pin its self-consistency."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
OBSERVABILITY = REPO / "observability"
COMPOSE = REPO / "docker-compose"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@pytest.fixture(scope="module")
def prometheus_config() -> dict:
    return _load_yaml(OBSERVABILITY / "prometheus" / "prometheus.yml")


@pytest.fixture(scope="module")
def datasource() -> dict:
    provisioned = _load_yaml(
        OBSERVABILITY / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
    )
    (provisioned_datasource,) = provisioned["datasources"]
    return provisioned_datasource


def _job(prometheus_config: dict, name: str) -> dict:
    (job,) = [j for j in prometheus_config["scrape_configs"] if j["job_name"] == name]
    return job


def test_prometheus_scrapes_exactly_the_three_agreed_targets(prometheus_config: dict) -> None:
    """The service, the broker, the GPU — and nothing else on purpose."""
    jobs = {job["job_name"] for job in prometheus_config["scrape_configs"]}
    assert jobs == {"serving", "redpanda", "dcgm"}


def test_redpanda_is_scraped_on_its_public_metrics_path(prometheus_config: dict) -> None:
    """/metrics (the default) is Redpanda's internal firehose; the dashboards read public."""
    assert _job(prometheus_config, "redpanda")["metrics_path"] == "/public_metrics"


def test_every_scrape_target_is_a_service_on_the_shared_network(prometheus_config: dict) -> None:
    """Static targets are compose DNS names — a rename there silently empties a dashboard."""
    services = set(_load_yaml(COMPOSE / "docker-compose.serving.yml")["services"]) | set(
        _load_yaml(COMPOSE / "docker-compose.streaming.yml")["services"]
    )
    for job in prometheus_config["scrape_configs"]:
        for static in job["static_configs"]:
            for target in static["targets"]:
                host = target.rsplit(":", 1)[0]
                assert host in services, f"scrape target {target!r} names no compose service"


def test_serving_scrape_port_matches_the_stack_env(prometheus_config: dict) -> None:
    """prometheus.yml cannot interpolate env vars, so the port is pinned here instead."""
    (static,) = _job(prometheus_config, "serving")["static_configs"]
    (target,) = static["targets"]
    env = (COMPOSE / ".env.serving").read_text()
    assert f"SERVING_PORT={target.rsplit(':', 1)[1]}" in env


def test_grafana_datasource_points_at_the_stack_prometheus(datasource: dict) -> None:
    assert datasource["type"] == "prometheus"
    assert datasource["url"] == "http://prometheus:9090"
    assert datasource["isDefault"] is True
    assert datasource["uid"]  # dashboards reference the datasource by this uid


def test_dashboards_parse_and_reference_the_provisioned_datasource(datasource: dict) -> None:
    dashboards = sorted((OBSERVABILITY / "grafana" / "dashboards").glob("*.json"))
    assert {p.stem for p in dashboards} == {"serving", "gpu", "streaming"}
    for path in dashboards:
        dashboard = json.loads(path.read_text())
        assert dashboard["panels"], f"{path.name} has no panels"
        for panel in dashboard["panels"]:
            assert panel["datasource"]["uid"] == datasource["uid"], (
                f"{path.name} panel {panel['title']!r} references a foreign datasource"
            )
            assert panel["targets"], f"{path.name} panel {panel['title']!r} queries nothing"


def test_dashboard_provider_path_matches_the_compose_mount() -> None:
    """Grafana loads dashboards from where compose actually mounts the committed JSON."""
    provisioned = _load_yaml(
        OBSERVABILITY / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
    )
    (provider,) = provisioned["providers"]
    assert provider["allowUiUpdates"] is False  # the files are the source of truth
    container_path = provider["options"]["path"]

    grafana = _load_yaml(COMPOSE / "docker-compose.serving.yml")["services"]["grafana"]
    mounts = {v.split(":")[1] for v in grafana["volumes"]}
    assert container_path in mounts


def test_streaming_init_enables_the_lag_gauges_the_dashboard_queries() -> None:
    """redpanda_kafka_consumer_group_lag_* exist only when the cluster config opts in."""
    init = _load_yaml(COMPOSE / "docker-compose.streaming.yml")["services"]["redpanda-init"]
    script = init["entrypoint"][-1]
    assert "enable_consumer_group_metrics" in script
    assert "consumer_lag" in script
