"""Unit tests for the producer's demo-store discovery + CLI defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlops_cv.config import Settings
from mlops_cv.streaming.producer import build_parser, discover_sequences, resolve_shift


def _store(root: Path, sequences: dict[str, list[int]]) -> Path:
    for sequence, indexes in sequences.items():
        seq_dir = root / "demo" / "images" / sequence
        seq_dir.mkdir(parents=True)
        for index in indexes:
            (seq_dir / f"{index:07d}.jpg").write_bytes(b"\xff\xd8fake")
    return root


def test_discovers_all_sequences_sorted_frames(tmp_path: Path) -> None:
    _store(tmp_path, {"seqB": [21, 1], "seqA": [41]})
    clips = discover_sequences(tmp_path, None)
    assert list(clips) == ["seqA", "seqB"]
    assert [p.stem for p in clips["seqB"]] == ["0000001", "0000021"]  # sorted = replay order


def test_sequence_filter_selects_and_validates(tmp_path: Path) -> None:
    _store(tmp_path, {"seqA": [1], "seqB": [1]})
    assert list(discover_sequences(tmp_path, ["seqB"])) == ["seqB"]
    with pytest.raises(SystemExit, match="unknown sequences.*seqC"):
        discover_sequences(tmp_path, ["seqC"])


def test_missing_demo_store_fails_with_ingest_hint(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="make ingest"):
        discover_sequences(tmp_path, None)


def test_empty_sequence_dirs_are_not_a_demo_store(tmp_path: Path) -> None:
    (tmp_path / "demo" / "images" / "seqA").mkdir(parents=True)  # dir exists, no frames
    with pytest.raises(SystemExit, match="make ingest"):
        discover_sequences(tmp_path, None)


def test_parser_defaults_come_from_settings(tmp_path: Path) -> None:
    settings = Settings(_env_file=str(tmp_path / "none"))  # type: ignore[call-arg]
    args = build_parser(settings).parse_args([])
    assert args.subset_dir == settings.data.subset_dir
    assert args.fps == settings.streaming.fps
    assert args.loops == 0  # forever
    assert args.sequences is None  # all clips
    assert args.shift is None  # no synthetic drift unless asked for
    assert args.shift_amount is None  # each mode carries its own measured default


def test_shift_flags_resolve_to_the_transform_the_producer_applies(tmp_path: Path) -> None:
    """The default replay must stay unshifted — byte-for-byte publishing depends on it."""
    from mlops_cv.streaming.shift import Shift

    parser = build_parser(Settings(_env_file=str(tmp_path / "none")))  # type: ignore[call-arg]

    assert Shift.resolve(*_shift_args(parser, [])) is None
    assert Shift.resolve(*_shift_args(parser, ["--shift", "defocus"])) == Shift(
        mode="defocus", amount=4.0
    )
    assert Shift.resolve(
        *_shift_args(parser, ["--shift", "brightness", "--shift-amount", "1.6"])
    ) == Shift(mode="brightness", amount=1.6)


def test_an_unusable_shift_is_refused_at_the_command_line(tmp_path: Path) -> None:
    """Every way of asking for a shift that cannot happen fails here, not mid-replay."""
    parser = build_parser(Settings(_env_file=str(tmp_path / "none")))  # type: ignore[call-arg]

    with pytest.raises(SystemExit):
        parser.parse_args(["--shift", "snow"])
    with pytest.raises(SystemExit):
        resolve_shift(parser, parser.parse_args(["--shift-amount", "0.5"]))
    with pytest.raises(SystemExit):
        resolve_shift(parser, parser.parse_args(["--shift", "defocus", "--shift-amount", "0"]))


def _shift_args(parser, argv: list[str]) -> tuple[str | None, float | None]:  # noqa: ANN001
    args = parser.parse_args(argv)
    return args.shift, args.shift_amount
