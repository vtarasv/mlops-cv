"""Application settings from a single ``.env`` file plus the OS environment (which wins).

Precedence (lowest first): ``.env``  <  OS environment.
Defaults the compose env files repeat are pinned by ``tests/unit/test_stack_constants.py``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mlops_cv.data.convert_visdrone_vid import CLASS_MERGE, RAW_NAMES, YOLO_NAMES


class DataSettings(BaseModel):
    """Dataset locations."""

    raw_dir: Path = Path("data/raw")
    subset_dir: Path = Path("data/visdrone-vid-small")
    dataset_yaml: Path = Path("configs/datasets/VisDrone-VID-merged.yaml")


class MlflowSettings(BaseModel):
    """MLflow tracking + registry coordinates."""

    tracking_uri: str = "http://localhost:5000"
    experiment: str = "aerial-object-detection"
    registered_model_prefix: str = "aerial-object-detector"
    champion_alias: str = "champion"  # registry alias marking the deployed model

    def registered_model(self, target: str) -> str:
        """Each deployment target's own registered model (own champion, own gate)."""
        return f"{self.registered_model_prefix}-{target}"


class TrainingSettings(BaseModel):
    """The shared YOLO26 recipe (its § Final recipe); the target picks the size.

    See docs/tuning/yolo26-visdrone-vid.md.
    """

    weights: str = "yolo26s.pt"  # its directory is the pretrained-weights cache
    epochs: int = 30
    imgsz: int = 1280
    batch: int = -1  # -1 = ultralytics autobatch; the resolved value is logged as a run param
    nbs: int = 64  # nominal batch size -> accumulate = round(nbs / batch) for the small VRAM budget
    patience: int = 20
    workers: int = 4
    cache: bool | str = False  # or "ram"/"disk": neither measured faster locally
    amp: bool = True
    device: str = "0"


class StreamingSettings(BaseModel):
    """Streaming-inference coordinates: broker, topics, consumer groups, pacing, anomaly rule."""

    bootstrap_servers: str = "localhost:19092"
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
    metrics_port: int = 9101  # scraped in-network


class OptimizeSettings(BaseModel):
    """Serving-variant production: the compiler's budget and the edge gate's tolerance."""

    workspace_gb: int = 4  # compiler workspace ceiling
    device_tolerance_pt: float = 0.5  # max |device - desktop| mAP gap, in percentage points


class ServingSettings(BaseModel):
    """The HTTP detection service: where it listens and how it answers by default."""

    host: str = "0.0.0.0"
    port: int = 8000
    conf_threshold: float = 0.25  # default detection confidence floor
    max_inflight: int = Field(default=2, ge=1)  # detects at once, per service or consumer


GDINO_B = "IDEA-Research/grounding-dino-base"
SAM3 = "facebook/sam3"
# The labeler's detectors.
DETECTORS = (GDINO_B, SAM3)


class AutolabelDetector(BaseModel):
    """The auto-labeler's text-prompted detector."""

    model_config = ConfigDict(extra="forbid")

    # or ``SAM3``, chosen with box_filter.nms_iou 0.5 and confidence_floor 0.593
    model: str = GDINO_B
    # prompt text -> the merged class its boxes get; VisDrone's 10 raw names
    prompts: dict[str, str] = Field(
        default_factory=lambda: {n: YOLO_NAMES[CLASS_MERGE[c]] for c, n in RAW_NAMES.items()}
    )
    threshold: float = 0.001  # the detector's own score floor (val's); it keeps its top 300
    dtype: Literal["bfloat16", "float16"] | None = "float16"  # ``None`` is fp32

    @field_validator("prompts")
    @classmethod
    def _merged_classes(cls, prompts: dict[str, str]) -> dict[str, str]:
        if unknown := set(prompts.values()) - set(YOLO_NAMES.values()):
            raise ValueError(f"not a merged class: {sorted(unknown)}")
        return prompts


class BoxFilter(BaseModel):
    """The box filter's stages, applied in this order; ``None`` turns a stage off."""

    model_config = ConfigDict(extra="forbid")

    min_side: float | None = 8  # px
    max_area: float | None = 0.05  # share of the frame
    min_score: float | None = None
    nms_iou: float | None = None  # class-agnostic
    nested_ios: float | None = None  # of two boxes this share nested, drop one
    nested_keep: Literal["outer", "inner"] = "outer"  # ... the box the nested-box rule keeps
    max_objects: int | None = None  # the highest-scoring K


# The box-tracker keys the confidence floor sets.
FLOOR_KEYS = frozenset({"track_high_thresh", "new_track_thresh"})


class AutolabelTracker(BaseModel):
    """An ultralytics box tracker over the box filter's output, fresh per clip."""

    model_config = ConfigDict(extra="forbid")

    model: str = "tracktrack"  # an ultralytics tracker YAML's stem
    overrides: dict[str, Any] = {}  # its YAML keys over ultralytics' defaults; none changed

    @field_validator("overrides")
    @classmethod
    def _floor_keys_are_the_floors(cls, overrides: dict[str, Any]) -> dict[str, Any]:
        if taken := overrides.keys() & FLOOR_KEYS:
            raise ValueError(f"the confidence floor sets {sorted(taken)}")
        return overrides


class AutolabelSettings(BaseModel):
    """The labeling config (docs/tuning/autolabel-visdrone-vid.md): the winner so far."""

    model_config = ConfigDict(extra="forbid")

    detector: AutolabelDetector = AutolabelDetector()
    box_filter: BoxFilter = BoxFilter()
    tracker: AutolabelTracker = AutolabelTracker()
    # the tracker's ``FLOOR_KEYS``: a box below it matches no track and starts none
    confidence_floor: float = 0.218
    max_det: int = 300  # the top boxes kept per frame: the YOLO validator's (ultralytics `max_det`)
    runs_dir: Path = Path("runs/autolabel")  # the FiftyOne App mounts it: HOST_AUTOLABEL_RUNS

    @field_validator("detector")
    @classmethod
    def _known_detector(cls, detector: AutolabelDetector) -> AutolabelDetector:
        if detector.model not in DETECTORS:
            raise ValueError(f"detector.model {detector.model!r} is not one of {DETECTORS}")
        return detector


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

    log_level: str = "INFO"
    data: DataSettings = DataSettings()
    mlflow: MlflowSettings = MlflowSettings()
    training: TrainingSettings = TrainingSettings()
    streaming: StreamingSettings = StreamingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    optimize: OptimizeSettings = OptimizeSettings()
    serving: ServingSettings = ServingSettings()
    autolabel: AutolabelSettings = AutolabelSettings()


def load_settings(base_dir: str | Path = ".") -> Settings:
    """Load settings from ``<base_dir>/.env`` plus the OS environment (which wins)."""
    return Settings(_env_file=str(Path(base_dir) / ".env"))  # type: ignore[call-arg]


@lru_cache
def get_settings() -> Settings:
    """Return cached process-wide settings."""
    return load_settings()
