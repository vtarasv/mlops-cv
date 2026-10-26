"""Live view: stats, variant → imgsz, the web handler, the CLI defaults, the camera pipe."""

from __future__ import annotations

import json
import subprocess
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

from mlops_cv.edge import live_view
from mlops_cv.startup import StartupError

NAMES = {0: "car", 1: "person"}
DETS = np.array(
    [[10, 10, 40, 30, 0.9, 0], [50, 20, 60, 50, 0.5, 0], [5, 40, 20, 60, 0.4, 1]], np.float32
)


def test_stats_count_boxes_per_class_name() -> None:
    s = live_view.frame_stats(DETS, NAMES, "m", 5.66, 171.4, 240.6)
    assert s == {
        "model": "m",
        "fps": 5.7,
        "infer_ms": 171,
        "frame_age_ms": 241,
        "counts": {"car": 2, "person": 1},
        "total": 3,
    }


def test_draw_boxes_a_copy_in_the_class_colour() -> None:
    frame = np.zeros((80, 80, 3), np.uint8)
    out = live_view.draw(frame, DETS, NAMES)
    assert not frame.any()  # the camera frame is left untouched
    assert tuple(out[20, 10]) == live_view.COLORS[0]  # left edge of the first box
    assert tuple(out[50, 5]) == live_view.COLORS[1]


def test_imgsz_comes_from_the_export_dir_name() -> None:
    assert live_view.imgsz_of(Path("models/ncnn-fp16-640_ncnn_model")) == 640
    assert live_view.imgsz_of(Path("ncnn-fp32-320_ncnn_model")) == 320


def test_parser_defaults() -> None:
    args = live_view.build_parser().parse_args([])
    assert args.show == "web"
    assert args.port == 8090  # 8000 is serving's
    assert args.model is None
    assert args.model_version is None  # champion
    assert args.variant is None  # the champion's served ncnn-fp16-<training imgsz>
    assert not args.pull_only
    assert (args.size, args.conf, args.label) == ("1280x720", 0.3, "")


@pytest.fixture
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), live_view.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_page_points_at_the_stream_and_stats(server: str) -> None:
    with urllib.request.urlopen(f"{server}/") as resp:
        body = resp.read()
    assert b'src="/stream.mjpg"' in body
    assert b"fetch('/stats')" in body


def test_stats_endpoint_serves_the_latest_frame_stats(server: str, monkeypatch) -> None:
    monkeypatch.setattr(live_view, "stats", {"fps": 5.7, "total": 2})
    with urllib.request.urlopen(f"{server}/stats?t=1") as resp:
        assert resp.headers["Content-Type"] == "application/json"
        assert json.loads(resp.read()) == {"fps": 5.7, "total": 2}


def test_unknown_path_is_404(server: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{server}/nope")
    assert exc.value.code == 404
    exc.value.close()  # the error holds the open response


def test_pull_only_prints_the_pulled_dir_and_never_opens_the_camera(
    monkeypatch, capsys, tmp_path
) -> None:
    pulled = tmp_path / "ncnn-fp16-640_ncnn_model"
    monkeypatch.setattr(live_view, "pull_model", lambda version, variant: pulled)
    monkeypatch.setattr(live_view, "open_camera", lambda size: pytest.fail("camera opened"))
    assert live_view.main(["--pull-only"]) == 0
    assert capsys.readouterr().out.strip() == str(pulled)


# 64x32 BGR in 8x8 solid blocks, so I420's 2x2 chroma subsampling barely changes it.
FRAME = np.kron(
    np.random.default_rng(0).integers(0, 256, (4, 8, 3), np.uint8), np.ones((8, 8, 1), np.uint8)
)


@pytest.fixture
def fake_rpicam(monkeypatch, tmp_path):
    """Swap rpicam-vid for ``cat`` of the I420 bytes the test writes."""
    raw = tmp_path / "frames.i420"
    popen = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: popen(["cat", raw], **kw))
    return raw


def i420(frame: np.ndarray) -> bytes:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).tobytes()


def test_camera_reads_bgr_frames_off_the_rpicam_pipe(fake_rpicam) -> None:
    fake_rpicam.write_bytes(i420(FRAME) * 3)
    cam = live_view.open_camera("64x32")
    try:
        frame = cam.capture_array()
        assert frame.shape == FRAME.shape
        assert np.abs(frame.astype(int) - FRAME).max() <= 3
    finally:
        cam.stop()


def test_no_camera_is_a_startup_error(fake_rpicam) -> None:
    fake_rpicam.write_bytes(b"")  # rpicam-vid exits before the first frame
    with pytest.raises(StartupError, match="rpicam-hello --list-cameras"):
        live_view.open_camera("64x32")


def test_a_short_read_mid_run_raises(fake_rpicam) -> None:
    whole = i420(FRAME)
    fake_rpicam.write_bytes(whole * 2 + whole[: len(whole) // 2])  # rpicam-vid died mid-frame
    cam = live_view.open_camera("64x32")
    try:
        cam.capture_array()
        with pytest.raises(EOFError):
            cam.capture_array()
    finally:
        cam.stop()


def test_a_width_rpicam_vid_would_pad_is_a_startup_error(fake_rpicam) -> None:
    with pytest.raises(StartupError, match="multiple of 64"):
        live_view.open_camera("1000x720")
