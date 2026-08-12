"""Entrypoint: resolve the champion's graph, load it, serve it."""

from __future__ import annotations

import argparse
import logging

from mlops_cv.config import Settings, get_settings
from mlops_cv.serving.app import create_app
from mlops_cv.serving.errors import StartupError
from mlops_cv.serving.resolve import champion_detector
from mlops_cv.tracking import client

logger = logging.getLogger(__name__)


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Serve the champion model's published graph over HTTP.")
    p.add_argument("--host", default=settings.serving.host)
    p.add_argument("--port", type=int, default=settings.serving.port)
    return p


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    args = build_parser(settings).parse_args(argv)

    client.configure(settings)
    try:
        detector = champion_detector(settings)
    except StartupError as exc:
        logger.error(str(exc))
        return 2

    import uvicorn

    logger.info(f"listening on {args.host}:{args.port}")
    uvicorn.run(create_app(detector, settings), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
