"""Replay the subset's demo store onto the frames topic as Frame messages.

The live-camera simulation of the streaming pipeline: cycles every demo-store clip,
publishing each JPEG byte-for-byte as it sits on disk (inline transport — no decode, no
re-encode), keyed by sequence with frame index + publish time in headers.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from pathlib import Path

from mlops_cv.config import Settings, get_settings
from mlops_cv.data.subset import DEMO_DIRNAME, IMAGES_DIRNAME
from mlops_cv.streaming.messages import FRAME_MAX_MESSAGE_BYTES, pack_frame

logger = logging.getLogger(__name__)

LOG_EVERY = 100  # frames between progress lines


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Publish demo-store frames as Frame messages.")
    p.add_argument(
        "--subset-dir",
        type=Path,
        default=settings.data.subset_dir,
        help="subset holding the demo store (default: DATA__SUBSET_DIR)",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=settings.streaming.fps,
        help="publish rate per second; 0 floods as fast as disk allows",
    )
    p.add_argument(
        "--loops",
        type=int,
        default=0,
        help="times to cycle the demo store; 0 loops forever",
    )
    p.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="demo-store sequences to publish (default: all)",
    )
    return p


def discover_sequences(subset_dir: Path, only: list[str] | None) -> dict[str, list[Path]]:
    """Demo-store clips as ``{sequence: sorted frame paths}``; fails with the run-ingest hint."""
    demo_images = subset_dir / DEMO_DIRNAME / IMAGES_DIRNAME
    found = sorted(demo_images.iterdir()) if demo_images.is_dir() else []
    clips = {d.name: frames for d in found if (frames := sorted(d.glob("*.jpg")))}
    if not clips:
        raise SystemExit(
            f"no demo store under {demo_images} — build it with `make ingest` "
            "(the ingest pipeline materializes the demo store into the subset)"
        )
    if only:
        unknown = sorted(set(only) - set(clips))
        if unknown:
            raise SystemExit(f"unknown sequences {unknown}; demo store has {sorted(clips)}")
        clips = {seq: clips[seq] for seq in only}
    return clips


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    args = build_parser(settings).parse_args(argv)

    # Discover before touching the broker so a missing demo store fails fast and clear.
    clips = discover_sequences(args.subset_dir, args.sequences)
    total_frames = sum(len(frames) for frames in clips.values())
    rate = f"{args.fps} fps" if args.fps > 0 else "flood rate"
    logger.info(f"publishing {len(clips)} clips / {total_frames} frames at {rate}")

    from confluent_kafka import Producer

    failures = 0

    def on_delivery(err, msg) -> None:
        nonlocal failures
        if err is not None:
            failures += 1
            logger.error(f"delivery failed for {msg.key()}: {err}")

    producer = Producer(
        {
            "bootstrap.servers": settings.streaming.bootstrap_servers,
            "enable.idempotence": True,  # broker dedups producer retries (acks=all implied)
            "message.max.bytes": FRAME_MAX_MESSAGE_BYTES,
            # Default delivery callback for every message (no per-message state to bind).
            "on_delivery": on_delivery,
        }
    )
    topic = settings.streaming.raw_frames_topic

    interval = 1.0 / args.fps if args.fps > 0 else 0.0
    next_due = time.perf_counter()
    published = 0
    loops = range(args.loops) if args.loops > 0 else itertools.count()
    try:
        for _ in loops:
            for sequence, frames in clips.items():
                for frame_path in frames:
                    key, headers = pack_frame(
                        sequence,
                        frame_index=int(frame_path.stem),
                        ts_ms=time.time_ns() // 1_000_000,
                    )
                    payload = frame_path.read_bytes()
                    while True:
                        try:
                            producer.produce(topic, payload, key=key, headers=headers)  # type: ignore[arg-type]
                            break
                        except BufferError:  # local queue full (flood mode): drain and retry
                            producer.poll(0.1)
                    producer.poll(0)  # serve delivery callbacks
                    published += 1
                    if published % LOG_EVERY == 0:
                        logger.info(f"published {published} frames")
                    if interval:
                        next_due += interval
                        time.sleep(max(0.0, next_due - time.perf_counter()))
    except KeyboardInterrupt:
        logger.info("interrupted; flushing")
    if undelivered := producer.flush(10):
        failures += undelivered
        logger.error(f"{undelivered} frames still undelivered after flush")
    logger.info(f"published {published} frames ({failures} delivery failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
