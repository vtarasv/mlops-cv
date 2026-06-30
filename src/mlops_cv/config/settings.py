"""Layered application settings with a local/dev/stage/prod config switch.

Precedence (lowest first): .env  <  .env.<env>  <  OS environment.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_VAR = "ENV"
DEFAULT_ENV = "local"


class Environment(StrEnum):
    local = "local"
    dev = "dev"
    stage = "stage"
    prod = "prod"


def current_env() -> str:
    return os.getenv(ENV_VAR, DEFAULT_ENV)


def env_files(env: str | None = None, base_dir: str | Path = ".") -> tuple[str, str]:
    """Return the layered env files, lowest precedence first: ``(.env, .env.<env>)``.

    pydantic-settings applies *later* files with *higher* precedence, so the
    environment overlay (``.env.<env>``) overrides the shared base (``.env``);
    OS environment variables override both. Missing files are ignored.
    """
    env = env or current_env()
    base = Path(base_dir)
    return (str(base / ".env"), str(base / f".env.{env}"))


class DataSettings(BaseModel):
    """Dataset locations."""

    raw_dir: Path = Path("data/raw")
    subset_dir: Path = Path("data/visdrone-vid-small")
    dataset_yaml: Path = Path("configs/datasets/VisDrone-VID-merged.yaml")


class MlflowSettings(BaseModel):
    """MLflow tracking + registry coordinates."""

    tracking_uri: str = "http://localhost:5000"
    experiment: str = "aerial-object-detection"
    registered_model: str = "aerial-object-detector"


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


class Settings(BaseSettings):
    """Process-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=env_files(),
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


def load_settings(env: str | None = None, base_dir: str | Path = ".") -> Settings:
    env = env or current_env()
    return Settings(env=env, _env_file=env_files(env, base_dir))  # type: ignore[call-arg]


@lru_cache
def get_settings() -> Settings:
    """Return cached settings for the active (``ENV``) environment."""
    return load_settings()
