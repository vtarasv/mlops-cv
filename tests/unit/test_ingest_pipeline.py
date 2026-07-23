"""Tests for the Beam ingestion pipeline on a tiny synthetic raw VisDrone-VID tree."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

beam = pytest.importorskip("apache_beam")

# Beam tears the Prism job server down via atexit — after pytest has closed its capture
# streams — and its "Really destroying service" WARNING then crashes the logging module
# ("I/O operation on closed file"). The shutdown is expected; keep that logger quiet.
logging.getLogger("apache_beam.utils.subprocess_server").setLevel(logging.ERROR)

# Aliased: a module-level name starting with "Test" would trip pytest's class collector.
from apache_beam.testing.test_pipeline import TestPipeline as BeamTestPipeline  # noqa: E402
from apache_beam.testing.util import assert_that, equal_to  # noqa: E402
from PIL import Image  # noqa: E402

from mlops_cv.data.subset import (  # noqa: E402
    DEMO_DIRNAME,
    MANIFEST_FIELDS,
    MANIFEST_FILENAME,
    RAW_SPLIT_DIRS,
    demo_store_present,
)
from mlops_cv.data.validate import validate_dataset  # noqa: E402
from mlops_cv.pipelines.ingest_pipeline import ConvertSequenceDoFn, main  # noqa: E402

STRIDE = 2
SIZE = (32, 24)


def _visdrone_line(
    frame: int, cat: int, left: int = 4, top: int = 4, w: int = 8, h: int = 8
) -> str:
    return f"{frame},0,{left},{top},{w},{h},1,{cat},0,0"


def _write_sequence(raw: Path, split: str, seq: str, n_frames: int, ann_lines: list[str]) -> None:
    seq_dir = raw / RAW_SPLIT_DIRS[split] / "sequences" / seq
    seq_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, n_frames + 1):
        Image.new("RGB", SIZE, (60 + 10 * idx, 90, 60)).save(seq_dir / f"{idx:07d}.jpg")
    ann_dir = raw / RAW_SPLIT_DIRS[split] / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / f"{seq}.txt").write_text("\n".join(ann_lines) + "\n", encoding="utf-8")


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    """Synthetic raw tree: 3 splits, classes {0,1,2} covered, strided + empty frames exercised."""
    root = tmp_path / "raw"
    # train seqA: 5 frames; boxes on 1 (person), 2 (vehicle), 4 (tricycle), 5 (person).
    # stride 2 keeps 1-based frames 1,3,5 -> frame 3 has no boxes -> kept rows: 1, 5.
    _write_sequence(
        root,
        "train",
        "seqA",
        5,
        [_visdrone_line(1, 1), _visdrone_line(2, 4), _visdrone_line(4, 3), _visdrone_line(5, 1)],
    )
    # val seqB: 2 frames; vehicle on frame 1 only -> kept rows: 1.
    _write_sequence(root, "val", "seqB", 2, [_visdrone_line(1, 4)])
    # test seqC (also the demo clip): 3 frames; tricycle on 1, person on 3 -> kept rows: 1, 3.
    _write_sequence(root, "test", "seqC", 3, [_visdrone_line(1, 3), _visdrone_line(3, 1)])
    return root


@pytest.fixture
def demo_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "demo_clips.yaml"
    p.write_text("split: test\nfps: 30\nmax_side: 1920\nclips:\n  - seqC\n", encoding="utf-8")
    return p


def test_convert_sequence_dofn_strides_and_skips_empty(raw: Path, tmp_path: Path) -> None:
    with BeamTestPipeline() as p:
        records = (
            p | beam.Create([("train", "seqA")]) | beam.ParDo(ConvertSequenceDoFn(str(raw), STRIDE))
        )
        base = raw / RAW_SPLIT_DIRS["train"] / "sequences" / "seqA"
        expected = [
            {
                "split": "train",
                "sequence": "seqA",
                "frame_index": idx,
                "image_relpath": f"images/train/seqA_{idx:07d}.jpg",
                "label_relpath": f"labels/train/seqA_{idx:07d}.txt",
                "width": SIZE[0],
                "height": SIZE[1],
                "n_boxes": 1,
                "n_person": 1,
                "n_vehicle": 0,
                "n_two_three_wheeler": 0,
                "src_image": str(base / f"{idx:07d}.jpg"),
                "label_text": "0 0.250000 0.333333 0.250000 0.333333\n",
            }
            for idx in (1, 5)  # stride 2 keeps 1,3,5; frame 3 has no surviving boxes
        ]
        assert_that(records, equal_to(expected))


def test_main_end_to_end(raw: Path, demo_yaml: Path, tmp_path: Path) -> None:
    out = tmp_path / "subset"
    template = Path("configs/datasets/VisDrone-VID-merged.yaml").resolve()
    assert (
        main(
            [
                f"--raw-dir={raw}",
                f"--output-dir={out}",
                f"--frame-stride={STRIDE}",
                f"--template-yaml={template}",
                f"--demo-clips={demo_yaml}",
                "--runner=DirectRunner",
            ]
        )
        == 0
    )

    # The subset is a valid dataset (layout, pairing, classes {0,1,2}, bbox ranges).
    validate_dataset(out)

    # Manifest: 2 train + 1 val + 2 test rows, standard schema, single shard.
    import csv

    with (out / MANIFEST_FILENAME).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == MANIFEST_FIELDS
    assert sorted((r["split"], r["sequence"], r["frame_index"]) for r in rows) == [
        ("test", "seqC", "1"),
        ("test", "seqC", "3"),
        ("train", "seqA", "1"),
        ("train", "seqA", "5"),
        ("val", "seqB", "1"),
    ]

    # Demo store: ALL 3 seqC frames at full rate; labels only for frames with boxes (1 and 3).
    assert demo_store_present(out)
    demo = out / DEMO_DIRNAME
    assert sorted(p.name for p in (demo / "images" / "seqC").glob("*.jpg")) == [
        "0000001.jpg",
        "0000002.jpg",
        "0000003.jpg",
    ]
    assert sorted(p.name for p in (demo / "labels" / "seqC").glob("*.txt")) == [
        "0000001.txt",
        "0000003.txt",
    ]
    # Byte-identical copy, no re-encode.
    src = raw / RAW_SPLIT_DIRS["test"] / "sequences" / "seqC" / "0000002.jpg"
    assert (demo / "images" / "seqC" / "0000002.jpg").read_bytes() == src.read_bytes()

    # Stamped dataset YAML with an absolute path.
    yaml = pytest.importorskip("yaml")
    stamped = yaml.safe_load((out / template.name).read_text(encoding="utf-8"))
    assert stamped["path"] == str(out.resolve())


def test_main_without_demo_clips(raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "subset"
    template = Path("configs/datasets/VisDrone-VID-merged.yaml").resolve()
    argv = [
        f"--raw-dir={raw}",
        f"--output-dir={out}",
        f"--frame-stride={STRIDE}",
        f"--template-yaml={template}",
        "--demo-clips=none",
        "--runner=DirectRunner",
    ]
    assert main(argv) == 0
    assert not (out / DEMO_DIRNAME / "images").exists()
    assert not demo_store_present(out)


def test_main_rejects_unknown_sequence(raw: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nope"):
        main([f"--raw-dir={raw}", f"--output-dir={tmp_path / 'x'}", "--sequences", "nope"])


def test_main_rejects_bad_stride(raw: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frame-stride"):
        main([f"--raw-dir={raw}", f"--output-dir={tmp_path / 'x'}", "--frame-stride=0"])


def test_missing_raw_split_raises_download_hint(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="docs/data.md"):
        main([f"--raw-dir={tmp_path / 'empty'}", f"--output-dir={tmp_path / 'x'}"])
