"""A live stream view: each frame with its detection event's boxes, drawn by the browser."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from mlops_cv.config import Settings, get_settings
from mlops_cv.data.convert_visdrone_vid import YOLO_NAMES
from mlops_cv.startup import StartupError, exits_on_startup_error
from mlops_cv.streaming.messages import (
    FRAME_MAX_MESSAGE_BYTES,
    DetectionEvent,
    FrameRef,
    parse_frame,
)

logger = logging.getLogger(__name__)

GROUP = "stream-view"

PENDING_CAP = 64  # frames awaiting their detection event; about 2 s at 30 fps
SEEN_CAP = 64  # sequence names remembered, so an older live sequence cannot take back over
EVENT_FIELDS = {"sequence", "frame_index", "ts_frame_ms", "latency_ms", "model", "boxes"}
STALE_MS = 1_000  # frame time without a followed frame before a seen sequence may take over
HEARTBEAT_S = 15.0  # an SSE comment this often finds a closed tab during a stall

# The newest event's JSON and its count, shared between the consumer loop and the web handler.
latest_event = ""
event_count = 0
event_ready = threading.Condition()


class Pairing:
    """Pairs raw frames with their detection events by frame address, in bounded memory."""

    def __init__(self, sequence: str | None = None) -> None:
        self.pinned = sequence
        self.sequence = sequence
        self.pending: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self.seen: OrderedDict[str, None] = OrderedDict()
        self.last: FrameRef | None = None  # the followed sequence's latest frame
        self.gaps = 0  # frames missing from the followed sequence's indices
        self.sessions = 0  # times the followed sequence changed, the first included

    def follow(self, ref: FrameRef) -> bool:
        """Whether ``ref``'s sequence is followed, switching to it when new or past a stall."""
        if ref.sequence == self.sequence and self.last is not None:
            return True
        if self.pinned is not None and ref.sequence != self.pinned:
            return False
        new = ref.sequence not in self.seen
        self.seen[ref.sequence] = None
        if len(self.seen) > SEEN_CAP:
            self.seen.popitem(last=False)
        stalled = self.last is None or ref.ts_ms - self.last.ts_ms > STALE_MS
        if not (new or stalled):
            return False
        self.sequence, self.last, self.gaps = ref.sequence, None, 0
        self.sessions += 1
        logger.info(f"following {ref.sequence}")
        self.pending.clear()
        return True

    def frame(self, ref: FrameRef, jpeg: bytes) -> None:
        if not self.follow(ref):
            return
        if self.last is not None and ref.frame_index > self.last.frame_index + 1:
            self.gaps += ref.frame_index - self.last.frame_index - 1
        self.last = ref
        self.pending[(ref.sequence, ref.frame_index)] = jpeg
        if len(self.pending) > PENDING_CAP:
            self.pending.popitem(last=False)

    def detection(self, event: DetectionEvent) -> tuple[DetectionEvent, bytes] | None:
        # A lookup, not a pop: a later duplicate event re-emits, so the later one wins.
        jpeg = self.pending.get((event.sequence, event.frame_index))
        return None if jpeg is None else (event, jpeg)


def event_json(event: DetectionEvent, jpeg: bytes, *, gaps: int, sessions: int) -> str:
    """One Server-Sent Event's data: the frame's JPEG as base64 beside its detection event."""
    return json.dumps(
        {
            **event.model_dump(include=EVENT_FIELDS),
            "jpeg_b64": base64.b64encode(jpeg).decode(),
            "gaps": gaps,
            "sessions": sessions,
        }
    )


def publish(data: str) -> None:
    """Hand ``data`` to every stream reader; a reader still busy skips to the newest."""
    global latest_event, event_count
    with event_ready:
        latest_event, event_count = data, event_count + 1
        event_ready.notify_all()


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Stream view</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
body{margin:0;background:#111;color:#eee;font:16px system-ui;display:flex;height:100vh}
.wrap{flex:1;display:flex;align-items:center;justify-content:center;min-width:0}
canvas{max-width:100%;max-height:100%}
aside{width:280px;padding:16px;background:#1b1b1b;font-variant-numeric:tabular-nums;overflow:auto}
aside h3{margin:14px 0 6px;font-size:14px;color:#999;text-transform:uppercase}
aside label,#panel div{display:flex;justify-content:space-between;gap:8px;padding:3px 0}
aside b{overflow-wrap:anywhere;text-align:right}
input[type=range]{width:100%}
.dead{color:#e55}
</style></head><body>
<div class=wrap><canvas id=view></canvas></div>
<aside><h3>Overlay</h3>
<label><span>boxes</span><input type=checkbox id=overlay checked></label>
<label><span>confidence &ge;</span><b id=confv></b></label>
<input type=range id=conf max=1 step=0.01>
<div id=classes></div><div id=panel></div></aside>
<script>
__CONFIG__;
const COLORS = ['#ff3838', '#3dd6ff', '#ffb21f'];
const $ = id => document.getElementById(id);
const canvas = $('view'), ctx = canvas.getContext('2d'), conf = $('conf');
conf.min = FLOOR; conf.value = FLOOR;
$('classes').innerHTML = CLASSES.map((c, i) =>
  `<label><span style="color:${COLORS[i % COLORS.length]}">${c}</span>` +
  `<input type=checkbox checked data-cls="${c}"></label>`).join('');
const hidden = new Set();
let ev = null, bmp = null, lost = false, arrivals = [], received = 0, shown = 0;

function draw() {
  $('confv').textContent = (+conf.value).toFixed(2);
  if (!bmp) return;
  const W = bmp.width, H = bmp.height;
  canvas.width = W; canvas.height = H; ctx.drawImage(bmp, 0, 0);
  if (!$('overlay').checked) return;
  ctx.lineWidth = Math.max(2, W / 640);
  ctx.font = `${Math.max(12, Math.round(W / 90))}px system-ui`;
  for (const b of ev.boxes) {
    if (b.conf < +conf.value || hidden.has(b.cls)) continue;
    const [x, y, w, h] = b.xywhn, left = (x - w / 2) * W, top = (y - h / 2) * H;
    ctx.strokeStyle = ctx.fillStyle = COLORS[Math.max(CLASSES.indexOf(b.cls), 0) % COLORS.length];
    ctx.strokeRect(left, top, w * W, h * H);
    ctx.fillText(`${b.cls} ${b.conf.toFixed(2)}`, left, top - 3);
  }
}

function el(tag, text, cls = '') {
  const e = document.createElement(tag);
  e.textContent = text; e.className = cls;
  return e;
}
function row(k, v, cls = '') {
  const d = document.createElement('div');
  d.append(el('span', k), el('b', v, cls));
  return d;
}
function panel() {
  const now = performance.now();
  arrivals = arrivals.filter(t => now - t < 1000);
  if (lost) return $('panel').replaceChildren(el('h3', 'Connection lost', 'dead'));
  if (!ev) return $('panel').replaceChildren(el('h3', 'Waiting for frames'));
  const age = Date.now() - ev.ts_frame_ms;
  $('panel').replaceChildren(
    el('h3', 'Stream'), row('sequence', ev.sequence),
    row('model', `${ev.model.name} v${ev.model.version}`),
    el('h3', 'Speed'), row('frames/s', arrivals.length),
    row('frame age', `${age} ms`, age > 1000 ? 'dead' : ''),
    row('inference', `${Math.round(ev.latency_ms)} ms`),
    el('h3', 'Link'), row('dropped frames', ev.gaps), row('sessions', ev.sessions));
}

for (const input of document.querySelectorAll('aside input')) input.oninput = () => {
  hidden.clear();
  for (const c of document.querySelectorAll('[data-cls]:not(:checked)')) hidden.add(c.dataset.cls);
  draw();
};
const events = new EventSource('/events');
events.onopen = () => { lost = false; panel(); };
events.onerror = () => { lost = true; panel(); };
events.onmessage = async e => {
  const n = ++received, next = JSON.parse(e.data);
  lost = false; arrivals.push(performance.now());
  const jpeg = await (await fetch('data:image/jpeg;base64,' + next.jpeg_b64)).blob();
  const img = await createImageBitmap(jpeg);
  if (n < shown) return img.close();  // a later frame decoded first
  shown = n; bmp?.close(); bmp = img; ev = next; draw();
};
draw(); setInterval(panel, 250);
</script></body></html>"""


def page(conf_floor: float, classes: Sequence[str]) -> bytes:
    """The page, with the slider's floor and the class checkboxes filled in."""
    config = f"const FLOOR={conf_floor},CLASSES={json.dumps(list(classes))}"
    return PAGE.replace("__CONFIG__", config).encode()


class Handler(BaseHTTPRequestHandler):
    page = b""  # set by main from the settings

    def log_message(self, format: str, *args: Any) -> None:  # one line per stream otherwise
        pass

    def do_GET(self) -> None:
        path = self.path.partition("?")[0]
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.page)))
            self.end_headers()
            self.wfile.write(self.page)
        elif path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.stream_events()
        else:
            self.send_error(404)

    def stream_events(self) -> None:
        seen = 0
        try:
            while True:
                with event_ready:
                    fresh = event_ready.wait_for(lambda seen=seen: event_count != seen, HEARTBEAT_S)
                    data, seen = latest_event, event_count
                self.wfile.write(f"data: {data}\n\n".encode() if fresh else b":\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", default=None, help="follow only this one (default: the newest)")
    p.add_argument("--port", type=int, default=8091, help="web port (8090: edge live view)")
    return p


def consumer_config(settings: Settings) -> dict:
    """A live tail that commits nothing, like the drift monitor's: every start shows now."""
    return {
        "bootstrap.servers": settings.streaming.bootstrap_servers,
        "group.id": GROUP,
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
        "fetch.message.max.bytes": FRAME_MAX_MESSAGE_BYTES,  # frames exceed the 1 MiB default
    }


def run_loop(consumer: Any, pairing: Pairing, frames_topic: str) -> None:
    """Frames and detection events in, one published pair per paired event, forever."""
    while True:
        msg = consumer.poll(0.2)
        if msg is None:
            continue
        if msg.error():
            logger.warning(f"consumer error: {msg.error()}")
            continue
        try:
            if (value := msg.value()) is None:
                raise ValueError("no value")
            if msg.topic() == frames_topic:
                pairing.frame(parse_frame(msg.key(), msg.headers()), value)
            elif pair := pairing.detection(DetectionEvent.from_value(value)):
                publish(event_json(*pair, gaps=pairing.gaps, sessions=pairing.sessions))
        except ValueError as exc:
            logger.warning(f"skipping a {msg.topic()} message: {exc}")


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    args = build_parser().parse_args(argv)
    streaming = settings.streaming

    from confluent_kafka import Consumer, KafkaException

    consumer = Consumer(consumer_config(settings))
    try:
        consumer.list_topics(timeout=10)
    except KafkaException as exc:
        raise StartupError(
            f"no broker at {streaming.bootstrap_servers} ({exc}); run `make serving-up`"
        ) from exc
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        raise StartupError(f"cannot listen on port {args.port} ({exc}); pass --port") from exc
    Handler.page = page(settings.serving.conf_threshold, list(YOLO_NAMES.values()))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    consumer.subscribe([streaming.raw_frames_topic, streaming.detections_topic])
    logger.info(f"stream view on http://localhost:{args.port}/ (group {GROUP}, commits nothing)")
    try:
        run_loop(consumer, Pairing(args.sequence), streaming.raw_frames_topic)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
