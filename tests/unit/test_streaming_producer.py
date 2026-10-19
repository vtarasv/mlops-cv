"""Unit tests for the producer's demo-store discovery + CLI defaults."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from mlops_cv.config import Settings
from mlops_cv.streaming.producer import (
    build_parser,
    discover_sequences,
    resolve_link,
    resolve_shift,
)


def _store(root: Path, sequences: dict[str, list[int]]) -> Path:
    for sequence, indexes in sequences.items():
        seq_dir = root / "demo" / "images" / sequence
        seq_dir.mkdir(parents=True)
        for index in indexes:
            (seq_dir / f"{index:07d}.jpg").write_bytes(b"\xff\xd8fake")
    return root


def _parser(tmp_path: Path) -> argparse.ArgumentParser:
    return build_parser(Settings(_env_file=str(tmp_path / "none")))  # type: ignore[call-arg]


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


def test_parser_defaults_leave_replay_flags_unset(tmp_path: Path) -> None:
    """``main`` fills them from settings, so ``--link`` can tell any explicit value apart."""
    args = _parser(tmp_path).parse_args([])
    assert (args.subset_dir, args.fps, args.loops, args.sequences) == (None, None, None, None)
    assert args.shift is None  # no synthetic drift unless asked for
    assert args.shift_amount is None  # each mode carries its own measured default


def test_shift_flags_resolve_to_the_transform_the_producer_applies(tmp_path: Path) -> None:
    """The default replay must stay unshifted — byte-for-byte publishing depends on it."""
    from mlops_cv.streaming.shift import Shift

    parser = _parser(tmp_path)

    assert _shift(parser, []) is None
    assert _shift(parser, ["--shift", "defocus"]) == Shift(mode="defocus", amount=4.0)
    assert _shift(parser, ["--shift", "brightness", "--shift-amount", "1.6"]) == Shift(
        mode="brightness", amount=1.6
    )


def test_an_unusable_shift_is_refused_at_the_command_line(tmp_path: Path) -> None:
    """Every way of asking for a shift that cannot happen fails here, not mid-replay."""
    parser = _parser(tmp_path)

    with pytest.raises(SystemExit):
        parser.parse_args(["--shift", "snow"])
    with pytest.raises(SystemExit):
        resolve_shift(parser, parser.parse_args(["--shift-amount", "0.5"]))
    with pytest.raises(SystemExit):
        resolve_shift(parser, parser.parse_args(["--shift", "defocus", "--shift-amount", "0"]))


def test_link_flags_default_to_demo_replay(tmp_path: Path) -> None:
    parser = _parser(tmp_path)
    args = parser.parse_args([])
    assert (args.link, args.link_name, args.link_timeout) == (None, "link", 5.0)
    assert resolve_link(parser, args) is None  # no link: replay the demo store


def test_link_options_parse_into_ffmpeg_input_options(tmp_path: Path) -> None:
    parser = _parser(tmp_path)
    argv = ["--link", "/dev/video0", "--link-format", "v4l2", "--shift", "defocus"]
    argv += ["--link-option", "video_size=1920x1080", "--link-option", "input_format=mjpeg"]
    args = parser.parse_args(argv)
    assert resolve_link(parser, args) == {"video_size": "1920x1080", "input_format": "mjpeg"}
    assert args.link_format == "v4l2"
    assert resolve_link(parser, parser.parse_args(["--link", "udp://0.0.0.0:5600"])) == {}


@pytest.mark.parametrize(
    "argv",
    [
        ["--link", "x.sdp", "--subset-dir", "/data"],  # the link sets its own pace and frames
        ["--link", "x.sdp", "--sequences", "seqA"],
        ["--link", "x.sdp", "--loops", "2"],
        ["--link", "x.sdp", "--fps", "0"],
        ["--link", "x.sdp", "--fps", "30"],  # the default value, passed explicitly
        ["--link", "x.sdp", "--loops", "0"],
        ["--link-format", "v4l2"],  # link flags without a link
        ["--link-option", "framerate=30"],
        ["--link", "/dev/video0", "--link-option", "framerate"],  # not key=value
    ],
)
def test_link_refuses_flags_it_cannot_honour(tmp_path: Path, argv: list[str]) -> None:
    parser = _parser(tmp_path)
    with pytest.raises(SystemExit):
        resolve_link(parser, parser.parse_args(argv))


def _shift(parser, argv: list[str]):  # noqa: ANN001
    return resolve_shift(parser, parser.parse_args(argv))
