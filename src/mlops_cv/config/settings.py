"""Layered application settings with a local/dev/stage/prod config switch.

Precedence (lowest first): .env  <  .env.<env>  <  OS environment.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

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


def load_settings(env: str | None = None, base_dir: str | Path = ".") -> Settings:
    env = env or current_env()
    return Settings(env=env, _env_file=env_files(env, base_dir))  # type: ignore[call-arg]


@lru_cache
def get_settings() -> Settings:
    """Return cached settings for the active (``ENV``) environment."""
    return load_settings()
