"""Publish demo-store JPEGs byte-for-byte, or a decoded video link, as Frame messages."""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from collections.abc import Iterator
from pathlib import Path

from mlops_cv.config import Settings, get_settings
from mlops_cv.data.subset import DEMO_DIRNAME, IMAGES_DIRNAME
from mlops_cv.streaming.link import READ_TIMEOUT_S, SourceFrame, link_frames
from mlops_cv.streaming.messages import FRAME_MAX_MESSAGE_BYTES, pack_frame
from mlops_cv.streaming.shift import DEFAULT_AMOUNTS, MODES, Shift

logger = logging.getLogger(__name__)

LOG_EVERY = 100  # frames between progress lines
REPLAY_FLAGS = ("subset_dir", "sequences", "loops", "fps")  # default None, so --link refuses any


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Publish demo-store or video-link frames as Frames.")
    p.add_argument(
        "--subset-dir",
        type=Path,
        default=None,
        help=f"subset holding the demo store (default: {settings.data.subset_dir})",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help=f"publish rate per second; 0 floods as fast as disk allows "
        f"(default: {settings.streaming.fps:g})",
    )
    p.add_argument(
        "--loops",
        type=int,
        default=None,
        help="times to cycle the demo store; 0 loops forever (default: 0)",
    )
    p.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="demo-store sequences to publish (default: all)",
    )
    p.add_argument(
        "--shift",
        choices=MODES,
        default=None,
        help="apply a SYNTHETIC photometric shift before publishing (default: publish as-is)",
    )
    p.add_argument(
        "--shift-amount",
        type=float,
        default=None,
        help="how much to shift (defocus: blur radius; brightness: multiplier). "
        f"Defaults are measured demo values: {DEFAULT_AMOUNTS}",
    )
    p.add_argument(
        "--link",
        default=None,
        help="decode a live video link instead of replaying the demo store: an .sdp file, "
        "udp://0.0.0.0:5600, rtsp://…, or a capture device",
    )
    p.add_argument("--link-name", default="link", help="the link's sequence prefix")
    p.add_argument(
        "--link-timeout",
        type=float,
        default=READ_TIMEOUT_S,
        help="seconds without data before the link is lost",
    )
    p.add_argument(
        "--link-format",
        default=None,
        help="FFmpeg input format, e.g. v4l2 or avfoundation (default: probe)",
    )
    p.add_argument(
        "--link-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="FFmpeg input option, repeatable, e.g. video_size=1920x1080 or input_format=mjpeg",
    )
    return p


def resolve_link(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> dict[str, str] | None:
    """The link's FFmpeg input options; ``None`` means demo replay. Refusals fail here."""
    if args.link is None:
        if args.link_format is not None or args.link_option:
            parser.error("--link-format/--link-option need a --link")
        return None
    if replay := [f for f in REPLAY_FLAGS if getattr(args, f) is not None]:
        flags = ", ".join(f"--{f.replace('_', '-')}" for f in replay)
        parser.error(f"--link sets its own pace and frames; drop {flags}")
    options = {}
    for pair in args.link_option or []:
        key, sep, value = pair.partition("=")
        if not (key and sep):
            parser.error(f"--link-option takes key=value, not {pair!r}")
        options[key] = value
    return options


def resolve_shift(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Shift | None:
    """The flags as a shift — ``None`` means unshifted replay; anything unusable fails here."""
    if args.shift is None:
        if args.shift_amount is not None:
            parser.error("--shift-amount needs a --shift mode; without one nothing is shifted")
        return None
    try:
        amount = DEFAULT_AMOUNTS[args.shift] if args.shift_amount is None else args.shift_amount
        return Shift(args.shift, amount)
    except ValueError as exc:
        parser.error(str(exc))


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


def replay(clips: dict[str, list[Path]], loops: int, fps: float) -> Iterator[SourceFrame]:
    """The demo store's frames, paced at ``fps`` (0 floods)."""
    interval = 1.0 / fps if fps > 0 else 0.0
    next_due = time.perf_counter()
    for _ in range(loops) if loops > 0 else itertools.count():
        for sequence, frames in clips.items():
            for frame_path in frames:
                yield SourceFrame(sequence, int(frame_path.stem), frame_path.read_bytes())
                if interval:
                    next_due += interval
                    time.sleep(max(0.0, next_due - time.perf_counter()))


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    parser = build_parser(settings)
    args = parser.parse_args(argv)
    shift = resolve_shift(parser, args)
    link_options = resolve_link(parser, args)

    source: Iterator[SourceFrame]
    if link_options is None:
        # Discover before touching the broker so a missing demo store fails fast and clear.
        clips = discover_sequences(args.subset_dir or settings.data.subset_dir, args.sequences)
        fps = settings.streaming.fps if args.fps is None else args.fps
        total_frames = sum(len(frames) for frames in clips.values())
        rate = f"{fps} fps" if fps > 0 else "flood rate"
        logger.info(f"publishing {len(clips)} clips / {total_frames} frames at {rate}")
        source = replay(clips, args.loops or 0, fps)
    else:
        logger.info(f"publishing video link {args.link} as {args.link_name}-<UTC start>")
        source = link_frames(
            args.link,
            name=args.link_name,
            timeout=args.link_timeout,
            fmt=args.link_format,
            options=link_options,
        )

    if shift is not None:
        logger.warning(
            f"applying a {shift.describe()} to every frame before publishing — "
            "simulated drift, not unseen data, and re-encoded rather than byte-for-byte"
        )

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

    published = dropped = sessions = 0
    last_sequence = None

    def progress() -> str:
        link = (
            f", dropped {dropped} damaged, sessions {sessions}" if link_options is not None else ""
        )
        return f"published {published} frames{link}"

    try:
        for sequence, frame_index, payload in source:
            if sequence != last_sequence:
                last_sequence, sessions = sequence, sessions + 1
            if payload is None:  # a damaged link frame; its index stays a gap on the topic
                dropped += 1
                continue
            key, headers = pack_frame(
                sequence, frame_index=frame_index, ts_ms=time.time_ns() // 1_000_000
            )
            if shift is not None:
                payload = shift.apply(payload)
            while True:
                try:
                    producer.produce(topic, payload, key=key, headers=headers)  # type: ignore[arg-type]
                    break
                except BufferError:  # local queue full (flood mode): drain and retry
                    producer.poll(0.1)
            producer.poll(0)  # serve delivery callbacks
            published += 1
            if published % LOG_EVERY == 0:
                logger.info(progress())
    except KeyboardInterrupt:
        logger.info("interrupted; flushing")
    if undelivered := producer.flush(10):
        failures += undelivered
        logger.error(f"{undelivered} frames still undelivered after flush")
    logger.info(f"{progress()} ({failures} delivery failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
