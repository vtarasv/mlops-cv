"""The video-link source: the drop rule, link sessions over an MPEG-TS file, the SDPs."""

from __future__ import annotations

import io
import itertools
import os
import re
import signal
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
from PIL import Image

from mlops_cv.streaming import link
from mlops_cv.streaming.link import link_frames, next_dropping

FRAMES = 15


def _mpegts(path: Path) -> Path:
    """A short H.264 MPEG-TS link recording: libx264, a 5-frame GOP, like an OpenHD feed."""
    rng = np.random.default_rng(0)
    with av.open(str(path), "w", format="mpegts") as out:
        stream = out.add_stream("libx264", rate=30, options={"g": "5", "preset": "ultrafast"})
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for _ in range(FRAMES):
            pixels = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            out.mux(stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")))
        out.mux(stream.encode())
    return path


def test_a_link_session_yields_every_frame_as_a_jpeg_and_the_reopen_starts_a_new_sequence(
    tmp_path: Path,
) -> None:
    url = str(_mpegts(tmp_path / "link.ts"))
    got = list(itertools.islice(link_frames(url, name="uav1"), FRAMES + 1))

    first, second = got[:FRAMES], got[FRAMES]
    sequences = {frame.sequence for frame in first}
    assert len(sequences) == 1
    assert re.fullmatch(r"uav1-\d{8}T\d{6}Z", sequences.pop())
    assert [frame.frame_index for frame in first] == list(range(FRAMES))
    for frame in first:
        assert frame.jpeg is not None
        with Image.open(io.BytesIO(frame.jpeg)) as im:
            assert (im.format, im.size) == ("JPEG", (64, 48))
    # EOF ended the session; a new one began
    assert second.sequence != first[0].sequence and second.frame_index == 0


def test_a_frame_the_decoder_refuses_is_a_dropped_frame_not_a_lost_link(tmp_path: Path) -> None:
    """A capture device's short buffer (a UVC webcam's, measured) is one dropped frame index."""
    frame_bytes = 64 * 48 * 2  # yuyv422
    raw = tmp_path / "capture.yuv"
    raw.write_bytes(bytes(4 * frame_bytes) + bytes(frame_bytes - 4))  # the demuxer cuts by size
    options = {"video_size": "64x48", "pixel_format": "yuyv422", "framerate": "30"}
    got = list(
        itertools.islice(link_frames(str(raw), name="cam", fmt="rawvideo", options=options), 5)
    )

    assert [frame.frame_index for frame in got] == [0, 1, 2, 3, 4]
    assert len({frame.sequence for frame in got}) == 1
    assert [frame.jpeg is None for frame in got] == [False, False, False, False, True]


def test_ctrl_c_while_waiting_on_the_link_stops_instead_of_reopening(monkeypatch) -> None:
    """PyAV's err_check drops a KeyboardInterrupt its open/demux wait was hit by."""
    monkeypatch.setattr(link, "OPEN_TIMEOUT_S", 1.0)
    monkeypatch.setattr(link.time, "sleep", lambda s: pytest.fail("reopened after Ctrl-C"))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))  # a port nothing sends to
        port = s.getsockname()[1]
    threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT)).start()
    with pytest.raises(KeyboardInterrupt):
        next(link_frames(f"udp://127.0.0.1:{port}", name="uav1"))


def _replay(frames: list[tuple[bool, int, bool]]) -> list[bool]:
    """Run the drop rule over stub (is_corrupt, pts, key_frame) frames: 30 fps, 90 kHz clock."""
    dropping, prev_pts, out = False, None, []
    for is_corrupt, pts, key_frame in frames:
        frame = SimpleNamespace(is_corrupt=is_corrupt, pts=pts, key_frame=key_frame, duration=3000)
        dropping = next_dropping(dropping, prev_pts, frame)
        prev_pts = pts
        out.append(dropping)
    return out


def test_a_clean_stream_drops_nothing() -> None:
    frames = [(False, i * 3000, i % 5 == 0) for i in range(12)]
    assert _replay(frames) == [False] * 12


def test_a_flagged_frame_drops_until_the_next_keyframe() -> None:
    frames = [(i == 2, i * 3000, i % 5 == 0) for i in range(12)]
    assert _replay(frames) == [False, False, True, True, True] + [False] * 7


def test_a_pts_jump_drops_until_the_next_keyframe() -> None:
    pts = [0, 3000, 9000, 12000, 15000, 18000]  # the frame at 6000 never came out
    frames = [(False, p, p % 15000 == 0) for p in pts]
    assert _replay(frames) == [False, False, True, True, False, False]


def test_a_keyframe_after_a_jump_is_kept() -> None:
    """A lost frame just before an IDR damages nothing: the IDR references no earlier frame."""
    frames = [(False, 0, True), (False, 3000, False), (False, 15000, True), (False, 18000, False)]
    assert _replay(frames) == [False, False, False, False]


@pytest.mark.parametrize(("codec", "decoder"), [("h264", "h264"), ("h265", "hevc")])
def test_the_shipped_sdp_files_parse_and_then_wait_for_the_link(
    codec: str, decoder: str, tmp_path: Path
) -> None:
    """A well-formed SDP opens its RTP port with the right decoder, then times out on silence."""
    shipped = Path(__file__).resolve().parents[2] / "configs" / "link" / f"rtp-{codec}-5600.sdp"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:  # 5600 may carry a link-sim
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1] // 2 * 2  # RTP takes an even port, RTCP the next
    sdp = tmp_path / "link.sdp"
    sdp.write_text(shipped.read_text().replace("m=video 5600 ", f"m=video {port} "))
    options = {"protocol_whitelist": "file,udp,rtp"}
    with av.open(str(sdp), options=options, timeout=0.5) as container:
        assert container.streams.video[0].codec_context.name == decoder
        with pytest.raises(av.error.ExitError):
            next(container.demux())
