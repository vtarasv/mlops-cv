"""Integration smoke for the streaming Compose stack (Redpanda + Console + topic init).

Requires the stack to be up::

    make streaming-up
"""

from __future__ import annotations

import pytest

from mlops_cv.config import get_settings

pytestmark = pytest.mark.docker

TIMEOUT = 5.0


@pytest.fixture(scope="module")
def topics() -> dict:
    """Cluster topic metadata, or a skip when the broker is unreachable."""
    from confluent_kafka.admin import AdminClient

    servers = get_settings().streaming.bootstrap_servers
    admin = AdminClient({"bootstrap.servers": servers})
    try:
        metadata = admin.list_topics(timeout=TIMEOUT)
    except Exception as exc:  # confluent_kafka.KafkaException on timeout/refused
        pytest.skip(f"streaming stack not reachable at {servers}: {exc}")
    return metadata.topics


def test_topics_exist_with_expected_partitions(topics: dict) -> None:
    streaming = get_settings().streaming
    assert set(topics) >= {
        streaming.raw_frames_topic,
        streaming.detections_topic,
        streaming.alerts_topic,
    }
    assert len(topics[streaming.raw_frames_topic].partitions) == 3
    assert len(topics[streaming.detections_topic].partitions) == 3
    assert len(topics[streaming.alerts_topic].partitions) == 1


def test_raw_frames_topic_has_raised_message_cap() -> None:
    from confluent_kafka.admin import AdminClient
    from confluent_kafka.admin._config import ConfigResource
    from confluent_kafka.admin._resource import ResourceType

    streaming = get_settings().streaming
    admin = AdminClient({"bootstrap.servers": streaming.bootstrap_servers})
    resource = ConfigResource(ResourceType.TOPIC, streaming.raw_frames_topic)
    config = admin.describe_configs([resource])[resource].result(timeout=TIMEOUT)
    assert int(config["max.message.bytes"].value) >= 4 * 1024 * 1024
