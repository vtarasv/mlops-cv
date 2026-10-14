"""Live view: the pure parts — stats, variant → imgsz, the web handler, the CLI defaults."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlops_cv.edge import live_view


def test_stats_count_boxes_per_class_name() -> None:
    boxes = [0.0, 0.0, 1.0]

    class Boxes(list):
        cls = boxes

    stub = SimpleNamespace(names={0: "car", 1: "person"}, boxes=Boxes(boxes))
    s = live_view.frame_stats(stub, "m", 5.66, 171.4, 240.6)
    assert s == {
        "model": "m",
        "fps": 5.7,
        "infer_ms": 171,
        "frame_age_ms": 241,
        "counts": {"car": 2, "person": 1},
        "total": 3,
    }


def test_imgsz_comes_from_the_export_dir_name() -> None:
    assert live_view.imgsz_of(Path("models/ncnn-fp16-640_ncnn_model")) == 640
    assert live_view.imgsz_of(Path("ncnn-fp32-320_ncnn_model")) == 320


def test_parser_defaults() -> None:
    args = live_view.build_parser().parse_args([])
    assert args.show == "web"
    assert args.port == 8090  # 8000 is serving's
    assert args.model is None
    assert args.model_version is None  # champion
    assert args.variant == "ncnn-fp16-640"
    assert not args.pull_only
    assert (args.size, args.conf, args.label) == ("1280x720", 0.3, "")


@pytest.fixture
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), live_view.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_page_points_at_the_stream_and_stats(server: str) -> None:
    body = urllib.request.urlopen(f"{server}/").read()
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


def test_pull_only_prints_the_pulled_dir_and_never_opens_the_camera(
    monkeypatch, capsys, tmp_path
) -> None:
    pulled = tmp_path / "ncnn-fp16-640_ncnn_model"
    monkeypatch.setattr(live_view, "pull_model", lambda version, variant: pulled)
    monkeypatch.setattr(live_view, "open_camera", lambda size: pytest.fail("camera opened"))
    assert live_view.main(["--pull-only"]) == 0
    assert capsys.readouterr().out.strip() == str(pulled)
