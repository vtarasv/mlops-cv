"""The observability config is code — pin its self-consistency."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
OBSERVABILITY = REPO / "observability"
COMPOSE = REPO / "docker-compose"
DASHBOARDS = OBSERVABILITY / "grafana" / "dashboards"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _exprs(panel: dict) -> list[str]:
    return [target["expr"] for target in panel["targets"]]


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


def test_prometheus_scrapes_exactly_the_four_agreed_targets(prometheus_config: dict) -> None:
    """The service, the drift monitor, the broker, the GPU — and nothing else on purpose."""
    jobs = {job["job_name"] for job in prometheus_config["scrape_configs"]}
    assert jobs == {"serving", "drift", "redpanda", "dcgm"}


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


def test_drift_scrape_port_matches_the_stack_env(prometheus_config: dict) -> None:
    """The monitor binds what compose tells it, so the literal target must track that value."""
    (static,) = _job(prometheus_config, "drift")["static_configs"]
    (target,) = static["targets"]
    env = (COMPOSE / ".env.serving").read_text()
    assert f"MONITOR_METRICS_PORT={target.rsplit(':', 1)[1]}" in env


def test_grafana_datasource_points_at_the_stack_prometheus(datasource: dict) -> None:
    assert datasource["type"] == "prometheus"
    assert datasource["url"] == "http://prometheus:9090"
    assert datasource["isDefault"] is True
    assert datasource["uid"]  # dashboards reference the datasource by this uid


def test_dashboards_parse_and_reference_the_provisioned_datasource(datasource: dict) -> None:
    dashboards = sorted(DASHBOARDS.glob("*.json"))
    assert {p.stem for p in dashboards} == {"serving", "gpu", "streaming", "drift"}
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


def test_lag_panels_exclude_the_group_that_commits_nothing() -> None:
    """The monitor never commits, so broker lag for it measures a bookmark that never moves."""
    from mlops_cv.config import Settings

    group = Settings().monitoring.group
    streaming = OBSERVABILITY / "grafana" / "dashboards" / "streaming.json"
    dashboard = json.loads(streaming.read_text())
    lag_targets = [
        target
        for panel in dashboard["panels"]
        for target in panel["targets"]
        if "consumer_group_lag" in target["expr"]
    ]

    assert lag_targets, "the streaming dashboard queries no consumer lag at all"
    for target in lag_targets:
        assert f'redpanda_group!="{group}"' in target["expr"], (
            f"lag query {target['expr']!r} would chart the non-committing {group!r} group"
        )


@pytest.fixture(scope="module")
def drift_dashboard() -> dict:
    return json.loads((DASHBOARDS / "drift.json").read_text())


def test_every_drift_statistic_is_charted_against_its_own_bar(drift_dashboard: dict) -> None:
    """A score without its bar is half a reading — proximity is the thing being watched."""
    from mlops_cv.pipelines.profiling import DRIFT_METRICS

    charted: set[str] = set()
    for panel in drift_dashboard["panels"]:
        scored = {
            found.group(1)
            for expr in _exprs(panel)
            if (found := re.search(r'drift_score\{statistic="([a-z_]+)"\}', expr))
        }
        for statistic in scored:  # the bar must be in *this* panel, not merely somewhere
            bar = f'drift_threshold{{statistic="{statistic}"}}'
            assert any(bar in expr for expr in _exprs(panel)), (
                f"panel {panel['title']!r} plots the {statistic!r} score with no bar beside it"
            )
        charted |= scored

    assert charted == set(DRIFT_METRICS)


def test_prediction_health_is_charted_with_no_threshold_and_says_why(
    drift_dashboard: dict,
) -> None:
    """On moving-platform aerial video these were measured not to separate drift."""
    health = (
        "prediction_detections_per_frame",
        "prediction_mean_confidence",
        "prediction_class_share",
    )
    panels = {
        metric: panel
        for panel in drift_dashboard["panels"]
        for metric in health
        if any(metric in expr for expr in _exprs(panel))
    }

    assert set(panels) == set(health), "every prediction-health reading is charted"
    assert len({panel["id"] for panel in panels.values()}) == len(health), (
        "the health readings share a panel — each is a different question about the predictions"
    )
    for panel in panels.values():
        steps = panel["fieldConfig"]["defaults"].get("thresholds", {}).get("steps", [])
        assert [step for step in steps if step["value"] is not None] == [], (
            f"prediction panel {panel['title']!r} draws a bar for a reading that is never judged"
        )
        assert "judged" in panel.get("description", ""), (
            f"prediction panel {panel['title']!r} does not say why it carries no threshold"
        )


def test_the_drift_dashboard_queries_only_readings_the_monitor_exports(
    drift_dashboard: dict,
) -> None:
    """The panels and the collectors are written apart; a renamed reading empties a panel."""
    from prometheus_client import CollectorRegistry

    from mlops_cv.monitoring.drift import WindowVerdict
    from mlops_cv.monitoring.metrics import MonitorMetrics
    from mlops_cv.monitoring.monitor import PredictionHealth

    registry = CollectorRegistry()
    metrics = MonitorMetrics(registry)
    metrics.observe_window(
        WindowVerdict(scores={"brightness": 0.4}, thresholds={"brightness": 2.6}, crossed=())
    )
    metrics.observe_episode()
    metrics.observe_skipped_frame()
    metrics.observe_unrecorded_episode()
    metrics.observe_prediction(
        PredictionHealth(
            n_events=1, detections_per_frame=3.0, mean_confidence=0.5, class_shares={"car": 1.0}
        )
    )
    exported = {sample.name for family in registry.collect() for sample in family.samples}

    queried = {
        name
        for panel in drift_dashboard["panels"]
        for expr in _exprs(panel)
        for name in re.findall(r"\b(?:drift|prediction)_[a-z_]+", expr)
    }
    assert queried, "the drift dashboard queries no monitor reading at all"
    assert queried <= exported, (
        f"the dashboard queries {sorted(queried - exported)}, which the monitor never exports"
    )


def test_streaming_init_enables_the_lag_gauges_the_dashboard_queries() -> None:
    """redpanda_kafka_consumer_group_lag_* exist only when the cluster config opts in."""
    init = _load_yaml(COMPOSE / "docker-compose.streaming.yml")["services"]["redpanda-init"]
    script = init["entrypoint"][-1]
    assert "enable_consumer_group_metrics" in script
    assert "consumer_lag" in script
