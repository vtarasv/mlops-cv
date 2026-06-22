"""Unit tests for the dataset validator (synthetic trees in tmp_path; no dataset, no GPU)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlops_cv.data.validate import DatasetValidationError, validate_dataset

# Default valid dataset: classes 0/1 in train, class 2 in val -> all three present, 1:1 paired.
_TRAIN = {"a": ["0 0.5 0.5 0.2 0.2"], "b": ["1 0.4 0.4 0.1 0.1"]}
_VAL = {"c": ["2 0.6 0.6 0.2 0.2"]}


def _make_dataset(
    root: Path,
    *,
    train: dict[str, list[str]] | None = None,
    val: dict[str, list[str]] | None = None,
) -> Path:
    """Write an ``images/labels`` x ``train/val`` YOLO tree (empty .jpg stubs + label files)."""
    for split, items in (
        ("train", _TRAIN if train is None else train),
        ("val", _VAL if val is None else val),
    ):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem, lines in items.items():
            (root / "images" / split / f"{stem}.jpg").write_bytes(b"")
            (root / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
    return root


def test_valid_dataset_passes(tmp_path: Path) -> None:
    report = validate_dataset(_make_dataset(tmp_path))
    assert report.n_images == 3
    assert report.n_boxes == 3
    assert report.splits == {"train": 2, "val": 1}
    assert report.class_counts == {0: 1, 1: 1, 2: 1}
    assert report.summary().startswith("OK:")


def test_out_of_range_class_id_raises(tmp_path: Path) -> None:
    # keep all 3 classes present so only the bad id is flagged
    _make_dataset(
        tmp_path,
        train={"a": ["0 0.5 0.5 0.2 0.2", "3 0.5 0.5 0.2 0.2"], "b": ["1 0.4 0.4 0.1 0.1"]},
    )
    with pytest.raises(DatasetValidationError, match="class id 3"):
        validate_dataset(tmp_path)


def test_unpaired_label_raises(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    (tmp_path / "labels" / "train" / "orphan.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    with pytest.raises(DatasetValidationError, match="without an image"):
        validate_dataset(tmp_path)


@pytest.mark.parametrize("split", ["train", "val"])
def test_empty_split_raises(tmp_path: Path, split: str) -> None:
    _make_dataset(tmp_path)
    for p in (tmp_path / "images" / split).iterdir():
        p.unlink()
    for p in (tmp_path / "labels" / split).glob("*.txt"):
        p.unlink()
    with pytest.raises(DatasetValidationError, match="no images found"):
        validate_dataset(tmp_path)


def test_out_of_range_bbox_raises(tmp_path: Path) -> None:
    _make_dataset(tmp_path, train={"a": ["0 0.5 0.5 1.5 0.2"], "b": ["1 0.4 0.4 0.1 0.1"]})
    with pytest.raises(DatasetValidationError, match=r"bbox out of \[0,1\]"):
        validate_dataset(tmp_path)


def test_missing_merged_class_raises(tmp_path: Path) -> None:
    # only classes 0 and 1 present anywhere -> class 2 (two-three-wheeler) absent
    _make_dataset(
        tmp_path,
        train={"a": ["0 0.5 0.5 0.2 0.2"], "b": ["1 0.4 0.4 0.1 0.1"]},
        val={"c": ["0 0.6 0.6 0.2 0.2"]},
    )
    with pytest.raises(DatasetValidationError, match="two-three-wheeler"):
        validate_dataset(tmp_path)
