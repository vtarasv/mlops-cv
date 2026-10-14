"""Live camera detection on the edge device: camera → NCNN model → boxes, on screen or the web."""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from mlops_cv.config import get_settings
from mlops_cv.edge.bench import pull_model
from mlops_cv.optimize.variants import Variant
from mlops_cv.startup import StartupError, exits_on_startup_error

logger = logging.getLogger(__name__)

# The newest annotated JPEG and its stats, shared between the loop and the web handler.
latest_jpeg: bytes = b""
stats: dict[str, Any] = {}
frame_ready = threading.Condition()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--show", choices=("web", "screen"), default="web")
    p.add_argument("--port", type=int, default=8090, help="web port (8000 is serving's)")
    p.add_argument("--model", type=Path, help="local NCNN model dir (default: pull --variant)")
    p.add_argument("--model-version", default=None, help="version to pull (default: champion)")
    p.add_argument("--variant", default="ncnn-fp16-640")
    p.add_argument("--pull-only", action="store_true", help="pull, print the model dir, exit")
    p.add_argument("--size", default="1280x720", help="camera capture WxH")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--label", default="", help="model name in the stats (default: the dir name)")
    return p


def imgsz_of(model_dir: Path) -> int:
    """The inference size an NCNN export dir (``<variant>_ncnn_model``) was built for."""
    return Variant.parse(model_dir.name.removesuffix("_ncnn_model")).imgsz


def frame_stats(
    result: Any, model: str, fps: float, infer_ms: float, frame_age_ms: float
) -> dict[str, Any]:
    """What the page's side panel and the screen overlay show for one annotated frame."""
    return {
        "model": model,
        "fps": round(fps, 1),
        "infer_ms": round(infer_ms),
        "frame_age_ms": round(frame_age_ms),
        "counts": dict(Counter(result.names[int(c)] for c in result.boxes.cls)),
        "total": len(result.boxes),
    }


def fatal_on_error(target: Callable[[], None]) -> None:
    """Run ``target`` on a daemon thread whose crash ends the process."""

    def run() -> None:
        try:
            target()
        except BaseException:
            traceback.print_exc()
            os._exit(1)

    threading.Thread(target=run, daemon=True).start()


def run_loop(cam: Any, model: Any, imgsz: int, conf: float, label: str, show: Callable) -> None:
    """Newest frame → predict → plot → ``show(annotated, stats)``, until ``show`` returns False."""
    latest: list[tuple[float, Any]] = []

    def capture() -> None:
        while True:
            latest[:] = [(time.time(), cam.capture_array())]

    fatal_on_error(capture)
    seen, t_prev, fps = 0.0, time.time(), 0.0
    while True:
        if not latest or latest[0][0] == seen:
            time.sleep(0.005)
            continue
        seen, frame = latest[0]
        t0 = time.time()
        result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        infer_ms = (time.time() - t0) * 1000
        annotated = result.plot()
        now = time.time()
        fps = 0.8 * fps + 0.2 / max(now - t_prev, 1e-6)
        t_prev = now
        if not show(annotated, frame_stats(result, label, fps, infer_ms, (now - seen) * 1000)):
            return


def publish(annotated: Any, frame: dict[str, Any]) -> bool:
    """The web output: JPEG-encode the frame and wake every stream reader."""
    import cv2

    global latest_jpeg, stats
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with frame_ready:
        latest_jpeg, stats = buf.tobytes(), frame
        frame_ready.notify_all()
    return True


def on_screen(annotated: Any, frame: dict[str, Any]) -> bool:
    """The screen output: the frame with a stats overlay; ``q`` quits."""
    import cv2

    counts = " ".join(f"{k}:{n}" for k, n in frame["counts"].items())
    lines = [
        frame["model"],
        f"{frame['fps']} frames/s  {frame['infer_ms']} ms  age {frame['frame_age_ms']} ms",
        f"{frame['total']} objects  {counts}",
    ]
    for i, line in enumerate(lines):
        org = (10, 30 + 30 * i)
        cv2.putText(annotated, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("live view", annotated)
    return cv2.waitKey(1) & 0xFF != ord("q")


PAGE = b"""<!doctype html><html><head><meta charset=utf-8><title>Live view</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
body{margin:0;background:#111;color:#eee;font:16px system-ui;display:flex;height:100vh}
.wrap{flex:1;display:flex;align-items:center;justify-content:center;min-width:0}
img{max-width:100%;max-height:100%;object-fit:contain}
aside{width:260px;padding:16px;background:#1b1b1b;font-variant-numeric:tabular-nums}
aside h3{margin:0 0 8px;font-size:14px;color:#999;text-transform:uppercase}
aside div{display:flex;justify-content:space-between;padding:3px 0}
.dead{color:#e55}
</style></head><body>
<div class=wrap><img src="/stream.mjpg"></div><aside id=panel></aside>
<script>
const p=document.getElementById('panel');
async function tick(){
 try{const s=await (await fetch('/stats')).json();
  p.innerHTML=`<h3>Model</h3><div><span>${s.model}</span></div><h3>Speed</h3>
   <div><span>frames/s</span><b>${s.fps}</b></div>
   <div><span>inference</span><b>${s.infer_ms} ms</b></div>
   <div><span>frame age</span><b>${s.frame_age_ms} ms</b></div><h3>Objects (${s.total})</h3>`+
   Object.entries(s.counts).map(([k,n])=>`<div><span>${k}</span><b>${n}</b></div>`).join('');
 }catch(e){p.innerHTML='<h3 class=dead>Connection lost</h3>';}
 setTimeout(tick,500);}
tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # one line per MJPEG frame otherwise
        pass

    def reply(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.partition("?")[0]
        if path == "/":
            self.reply(PAGE, "text/html; charset=utf-8")
        elif path == "/stats":
            self.reply(json.dumps(stats).encode(), "application/json")
        elif path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    with frame_ready:
                        frame_ready.wait()
                        jpg = latest_jpeg
                    self.wfile.write(
                        b"--FRAME\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                        + jpg
                        + b"\r\n"
                    )
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def open_camera(size: str) -> Any:
    """The started camera at ``size`` (WxH), BGR frames; StartupError when none is attached."""
    from picamera2 import Picamera2  # pyright: ignore[reportMissingImports]  # apt-only, on the Pi

    if not Picamera2.global_camera_info():
        raise StartupError(
            "no camera found — check the ribbon cable with `rpicam-hello --list-cameras`"
        )
    width, height = (int(v) for v in size.split("x"))
    cam = Picamera2()
    # RGB888 lands in memory as BGR, exactly what OpenCV and ultralytics expect.
    cam.configure(
        cam.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
    )
    cam.start()
    return cam


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=get_settings().log_level.upper(), format="%(message)s")
    args = build_parser().parse_args(argv)
    model_dir = args.model or pull_model(args.model_version, args.variant).resolve()
    if args.pull_only:
        print(model_dir)
        return 0

    cam = open_camera(args.size)
    from ultralytics import YOLO

    model = YOLO(str(model_dir), task="detect")
    label = args.label or model_dir.name
    loop = (cam, model, imgsz_of(model_dir), args.conf, label)
    try:
        if args.show == "screen":
            run_loop(*loop, on_screen)
        else:
            fatal_on_error(lambda: run_loop(*loop, publish))
            logger.info(f"live view of {label} on http://0.0.0.0:{args.port}/")
            ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
