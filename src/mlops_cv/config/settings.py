"""Application settings from a single ``.env`` file plus the OS environment (which wins).

Precedence (lowest first): ``.env``  <  OS environment. ``settings.env`` is a plain
``local|dev|stage|prod`` identifier read from the ``ENV`` variable
(for environment-aware branching / run labels).
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    local = "local"
    dev = "dev"
    stage = "stage"
    prod = "prod"


class DataSettings(BaseModel):
    """Dataset locations."""

    raw_dir: Path = Path("data/raw")
    subset_dir: Path = Path("data/visdrone-vid-small")
    dataset_yaml: Path = Path("configs/datasets/VisDrone-VID-merged.yaml")


class MlflowSettings(BaseModel):
    """MLflow tracking + registry coordinates."""

    # Port MUST match MLFLOW_PORT in docker-compose/.env.mlflow (server publishes there).
    tracking_uri: str = "http://localhost:5000"
    experiment: str = "aerial-object-detection"
    registered_model: str = "aerial-object-detector"
    champion_alias: str = "champion"  # registry alias marking the deployed model


class TrainingSettings(BaseModel):
    """YOLO26s transfer-learning recipe (tuned for an 8 GB Blackwell GPU)."""

    weights: str = "yolo26s.pt"
    epochs: int = 30
    imgsz: int = 640
    batch: int = -1  # -1 = ultralytics autobatch; the resolved value is logged as a run param
    nbs: int = 64  # nominal batch size -> accumulate = round(nbs / batch) for the small VRAM budget
    patience: int = 20
    workers: int = 4
    cache: str = "disk"
    amp: bool = True
    device: str = "0"


class StreamingSettings(BaseModel):
    """Streaming-inference coordinates: broker, topics, consumer groups, pacing, anomaly rule."""

    # Port MUST match KAFKA_HOST_PORT in docker-compose/.env.streaming (broker publishes there).
    bootstrap_servers: str = "localhost:19092"
    # Topics MUST match *_TOPIC in docker-compose/.env.streaming
    raw_frames_topic: str = "raw-frames"
    detections_topic: str = "detections"
    alerts_topic: str = "alerts"
    inference_group: str = "inference-consumer"
    anomaly_group: str = "anomaly-consumer"
    fps: float = 30.0  # producer pacing (frames/s); the CLI's --fps 0 floods instead
    commit_interval_s: float = 5.0  # how often the inference consumer commits stored offsets
    anomaly_class: str = "person"  # class the windowed count rule watches
    anomaly_window_s: float = 5.0  # sliding window length (frame-timestamp time)
    anomaly_threshold: float = 30.0  # windowed mean count that opens an alert episode


class MonitoringSettings(BaseModel):
    """The drift monitor: what it samples, how it windows, and where it exposes readings."""

    group: str = "drift-monitor"  # consumer group
    sample_every: int = Field(default=5, ge=1)  # profile every k-th frame
    window_frames: int = Field(default=100, ge=1)  # sampled frames per tumbling window
    consecutive_windows: int = Field(default=2, ge=1)  # drifted windows that open an episode
    threshold_margin: float = Field(default=1.0, gt=0)  # multiplies the derived thresholds
    metrics_port: int = 9101  # scraped in-network; MUST match MONITOR_METRICS_PORT in .env.serving


class OptimizeSettings(BaseModel):
    """Serving-variant production: the two deployment targets and the compiler's budget."""

    server_imgsz: int = 640
    edge_imgsz: int = 320  # the low-resolution rung of the edge (NCNN) ladder
    workspace_gb: int = 4  # compiler workspace ceiling


class ServingSettings(BaseModel):
    """The HTTP detection service: where it listens and how it answers by default."""

    host: str = "0.0.0.0"
    port: int = 8000
    conf_threshold: float = 0.25  # default detection confidence floor


class Settings(BaseSettings):
    """Process-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        use_enum_values=True,
        extra="ignore",
    )

    env: Environment = Environment.local
    log_level: str = "INFO"
    data: DataSettings = DataSettings()
    mlflow: MlflowSettings = MlflowSettings()
    training: TrainingSettings = TrainingSettings()
    streaming: StreamingSettings = StreamingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    optimize: OptimizeSettings = OptimizeSettings()
    serving: ServingSettings = ServingSettings()


def load_settings(base_dir: str | Path = ".") -> Settings:
    """Load settings from ``<base_dir>/.env`` plus the OS environment (which wins)."""
    return Settings(_env_file=str(Path(base_dir) / ".env"))  # type: ignore[call-arg]


@lru_cache
def get_settings() -> Settings:
    """Return cached process-wide settings."""
    return load_settings()
