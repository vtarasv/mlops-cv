"""The streaming wire contract: Frame messages, Detection events, Alerts.

Single owner of every payload that crosses the streaming bus (the Container-contract
precedent): each payload's emit and parse live here, pinned together by unit tests, so no
process hand-rolls its own encoding. Pure pydantic/stdlib — importable everywhere.

A **Frame message** is deliberately not JSON (inline-transport): the value is the
JPEG bytes verbatim, the *key* is the sequence (per-clip ordering within a partition), and
the small identifying metadata rides in headers as human-readable ASCII decimals.
Detection events and Alerts are versioned JSON values, keyed by sequence too, so they stay
co-partitioned with their frames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self

from pydantic import BaseModel

# Header names of a Frame message (values: ASCII decimal strings, readable in Console).
FRAME_INDEX_HEADER = "frame_index"
TS_MS_HEADER = "ts_ms"

# Client-side size limit for Frame messages; must clear the raw-frames topic's
# max.message.bytes (the compose env pins the same value broker-side).
FRAME_MAX_MESSAGE_BYTES = 4 * 1024 * 1024

Headers = list[tuple[str, bytes]]
# What a *consumed* message may hand back: the Kafka client shares one header type between
# its produce and consume sides, so it also admits a mapping, str values and None values.
InboundHeaders = Mapping[str, str | bytes | None] | Sequence[tuple[str, str | bytes | None]]


class FrameRef(BaseModel):
    """The identity of one Frame message: which clip, which frame, published when."""

    sequence: str
    frame_index: int
    ts_ms: int  # producer publish time (epoch ms)


def pack_frame(sequence: str, *, frame_index: int, ts_ms: int) -> tuple[bytes, Headers]:
    """The (key, headers) pair of a Frame message; the value is the JPEG bytes themselves."""
    return sequence.encode(), [
        (FRAME_INDEX_HEADER, str(frame_index).encode()),
        (TS_MS_HEADER, str(ts_ms).encode()),
    ]


def parse_frame(key: bytes | None, headers: InboundHeaders | None) -> FrameRef:
    """Recover a :class:`FrameRef` from a consumed message's key + headers.

    Unknown headers are ignored (skew tolerance); a missing key, or a missing or empty
    required header, is a contract violation and raises ``ValueError``.
    """
    if not key:
        raise ValueError("frame message has no key (expected the sequence)")
    pairs = headers.items() if isinstance(headers, Mapping) else (headers or ())
    named = {name: value for name, value in pairs}
    try:
        frame_index, ts_ms = named[FRAME_INDEX_HEADER], named[TS_MS_HEADER]
    except KeyError as missing:
        raise ValueError(f"frame message is missing header {missing}") from None
    if frame_index is None or ts_ms is None:
        raise ValueError("frame message has an empty frame_index/ts_ms header")
    return FrameRef(sequence=key.decode(), frame_index=int(frame_index), ts_ms=int(ts_ms))


class Box(BaseModel):
    """One detected box: merged class name, confidence, normalized xywh (resolution-free)."""

    cls: str
    conf: float
    xywhn: tuple[float, float, float, float]


class ModelInfo(BaseModel):
    """Which model answered: registered name + version."""

    name: str
    version: str


class _KeyedJson(BaseModel):
    """A JSON value keyed by its ``sequence`` — co-partitioned with the frames it is about."""

    sequence: str

    def to_value(self) -> bytes:
        """The Kafka message value: compact single-line JSON."""
        return self.model_dump_json().encode()

    @classmethod
    def from_value(cls, value: bytes) -> Self:
        """Parse a consumed value; unknown fields are ignored (skew tolerance)."""
        return cls.model_validate_json(value)

    def kafka_key(self) -> bytes:
        """The Kafka message key: the sequence."""
        return self.sequence.encode()


class DetectionEvent(_KeyedJson):
    """One model answer for one Frame message — the record downstream readers consume."""

    schema_version: int = 1
    frame_index: int
    ts_frame_ms: int  # the frame's publish time (scene time under pacing)
    ts_infer_ms: int  # when inference finished (epoch ms)
    latency_ms: float
    model: ModelInfo
    boxes: list[Box]


class Alert(_KeyedJson):
    """One anomaly-rule episode opening: the windowed reading that crossed its threshold."""

    schema_version: int = 1
    rule: str  # which rule fired (e.g. "windowed-count")
    cls: str  # the class the rule watches
    window_s: float
    threshold: float
    observed: float  # the windowed reading at the crossing
    n_frames: int  # frames inside the window at the crossing
    frame_index: int  # with ``sequence``: the frame that tipped the window
    ts_ms: int  # that frame's scene time (its ts_frame_ms)
