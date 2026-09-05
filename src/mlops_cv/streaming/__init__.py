"""Streaming: the bus's wire contract, producer, consumers, and their shared CLI."""

from __future__ import annotations

import argparse


def consumer_parser(description: str, unit: str) -> argparse.ArgumentParser:
    """The options every consumer takes: where a new group starts, and the test/smoke bounds."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--offset-reset",
        choices=("latest", "earliest"),
        default="latest",
        help="where a NEW consumer group starts: latest = live tail, earliest = replay",
    )
    p.add_argument(
        "--max-messages", type=int, default=None, help=f"stop after this many {unit} (tests/smokes)"
    )
    p.add_argument(
        "--idle-timeout-s",
        type=float,
        default=None,
        help=f"stop after this long with no {unit} (tests/smokes)",
    )
    return p
