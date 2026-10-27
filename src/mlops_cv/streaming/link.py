"""Decode a drone's live video link into Frame-message payloads, one link session at a time."""

from __future__ import annotations

import io
import logging
import signal
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from types import FrameType
from typing import TYPE_CHECKING, NamedTuple

from mlops_cv.streaming.shift import JPEG_QUALITY

if TYPE_CHECKING:
    from av import VideoFrame

logger = logging.getLogger(__name__)

OPEN_TIMEOUT_S = 30.0
READ_TIMEOUT_S = 5.0  # no data this long is a lost link
REOPEN_WAIT_S = 1.0
RTP_BUFFER_BYTES = 4 * 1024 * 1024


class SourceFrame(NamedTuple):
    """One frame for the publish loop; a dropped link frame has no JPEG."""

    sequence: str
    frame_index: int
    jpeg: bytes | None


def next_dropping(dropping: bool, prev_pts: int | None, frame: VideoFrame) -> bool:
    """Drop from a flagged frame or a pts jump until the next keyframe; returns the next state."""
    gap = frame.pts - prev_pts if prev_pts is not None and frame.pts is not None else 0
    jump = bool(frame.duration) and gap > frame.duration  # no duration: no jump to measure
    return frame.is_corrupt or (not frame.key_frame and (dropping or jump))


def link_frames(
    url: str,
    *,
    name: str,
    timeout: float = READ_TIMEOUT_S,
    fmt: str | None = None,
    options: dict[str, str] | None = None,
) -> Iterator[SourceFrame]:
    """The link's frames forever, one link session per sequence, reopening on loss."""
    from av.error import FFmpegError

    options = dict(options or {})
    if url.endswith(".sdp"):
        options.setdefault("protocol_whitelist", "file,udp,rtp")
        # RTP reads its socket inline, so the kernel's default buffer overflows after the probe.
        options.setdefault("buffer_size", str(RTP_BUFFER_BYTES))
    # A Ctrl-C during an open/demux wait lands only when the deadline fires, inside PyAV's
    # err_check, which drops it (exceptval check=False) and raises ExitError: note it ourselves.
    interrupted = threading.Event()
    previous = signal.getsignal(signal.SIGINT)
    hooked = callable(previous) and threading.current_thread() is threading.main_thread()
    if hooked:

        def on_sigint(signum: int, frame: FrameType | None) -> None:
            interrupted.set()
            previous(signum, frame)

        signal.signal(signal.SIGINT, on_sigint)
    try:
        while True:
            try:
                yield from _session(url, name, fmt, options, timeout)
                reason = "stream ended"
            except FFmpegError as exc:
                if interrupted.is_set():
                    raise KeyboardInterrupt from exc
                reason = str(exc)
            logger.warning(f"link lost ({reason}); reopening in {REOPEN_WAIT_S:g} s")
            time.sleep(REOPEN_WAIT_S)
    finally:
        if hooked:
            signal.signal(signal.SIGINT, previous)


def _session(
    url: str, name: str, fmt: str | None, options: dict[str, str], timeout: float
) -> Iterator[SourceFrame]:
    import av
    from av.error import FFmpegError

    with av.open(url, format=fmt, options=options, timeout=(OPEN_TIMEOUT_S, timeout)) as container:
        stream = container.streams.video[0]
        sequence, index, dropping, prev_pts = "", 0, False, None
        for packet in container.demux(stream):  # a demux error is the lost link
            try:
                frames: list[VideoFrame | None] = list(stream.decode(packet))
            except FFmpegError:  # a packet the decoder refuses (a short capture buffer) is damage
                frames = [None]
            for frame in frames:
                if index == 0:  # the session's identity is its first frame's time
                    sequence = f"{name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
                if frame is None:
                    dropping, jpeg = True, None
                else:
                    dropping, prev_pts = next_dropping(dropping, prev_pts, frame), frame.pts
                    jpeg = None if dropping else _jpeg(frame)
                yield SourceFrame(sequence, index, jpeg)
                index += 1


def _jpeg(frame: VideoFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_image().save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()
