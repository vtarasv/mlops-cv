"""Promote a training run's model version to champion (a servable gate winner only)."""

from __future__ import annotations

import argparse
import logging

from mlops_cv.config import get_settings
from mlops_cv.evaluation.gate import promote
from mlops_cv.orchestration.handoff import best_weights_uri
from mlops_cv.startup import StartupError, exits_on_startup_error
from mlops_cv.tracking.resolve import registered_version


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True, help="the model version's training run")
    args = p.parse_args(argv)

    name = settings.mlflow.registered_model
    version = registered_version(best_weights_uri(args.run_id), name)
    if version is None:
        raise StartupError(f"run {args.run_id} registered no '{name}' version")
    promote(settings, version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
