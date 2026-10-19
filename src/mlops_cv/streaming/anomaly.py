"""The episode-scoped windowed-count rule (pure logic — the anomaly consumer wraps it).

Watches one class's per-frame count through a sliding window over **frame time** (scene
time — flood-mode replay must not distort episodes) and opens an Alert exactly once when
a sequence's windowed mean crosses the threshold, re-arming only after it drops back. Duplicate
Detection events (at-least-once replay) are idempotently absorbed: the window keys
entries by ``(sequence, frame_index)``, so a re-delivered frame replaces itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mlops_cv.streaming.messages import Alert, DetectionEvent

RULE_NAME = "windowed-count"


@dataclass
class EpisodeRule:
    """Sliding-window mean count of one class, with fire-once/re-arm episode semantics."""

    cls: str
    window_s: float
    threshold: float
    _window: dict[tuple[str, int], tuple[int, int]] = field(default_factory=dict)
    _watermark_ms: int = 0  # newest frame time seen
    _active: dict[str, bool] = field(default_factory=dict)  # per sequence: an episode is open

    def observe(self, event: DetectionEvent) -> Alert | None:
        """Fold one Detection event into the window; return an Alert if an episode opens."""
        count = sum(1 for box in event.boxes if box.cls == self.cls)
        # Keyed by frame address: an at-least-once duplicate replaces, never double-counts.
        self._window[(event.sequence, event.frame_index)] = (event.ts_frame_ms, count)
        self._watermark_ms = max(self._watermark_ms, event.ts_frame_ms)
        horizon = self._watermark_ms - int(self.window_s * 1000)
        self._window = {key: (ts, n) for key, (ts, n) in self._window.items() if ts >= horizon}

        live = {seq for seq, _ in self._window}
        self._active = {seq: on for seq, on in self._active.items() if seq in live}

        counts = [n for (seq, _), (_, n) in self._window.items() if seq == event.sequence]
        if not counts:  # the frame itself fell behind the shared watermark
            return None
        mean = sum(counts) / len(counts)
        crossed = mean > self.threshold
        opened = crossed and not self._active.get(event.sequence, False)
        self._active[event.sequence] = crossed
        if not opened:
            return None
        return Alert(
            rule=RULE_NAME,
            cls=self.cls,
            window_s=self.window_s,
            threshold=self.threshold,
            observed=mean,
            n_frames=len(counts),
            sequence=event.sequence,
            frame_index=event.frame_index,
            ts_ms=event.ts_frame_ms,
        )
