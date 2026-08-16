"""Tests for the Beam profiling pipeline on a tiny synthetic subset."""

from __future__ import annotations

import csv
import json
import logging
import statistics
from pathlib import Path

import pytest

beam = pytest.importorskip("apache_beam")

# Beam tears the Prism job server down via atexit — after pytest has closed its capture
# streams — and its "Really destroying service" WARNING then crashes the logging module
# ("I/O operation on closed file"). The shutdown is expected; keep that logger quiet.
logging.getLogger("apache_beam.utils.subprocess_server").setLevel(logging.ERROR)

from PIL import Image  # noqa: E402

from mlops_cv.data.subset import (  # noqa: E402
    MANIFEST_FILENAME,
    manifest_csv_line,
    manifest_header,
)
from mlops_cv.pipelines.profile_pipeline import StatsCombineFn, main  # noqa: E402
from mlops_cv.pipelines.profiling import (  # noqa: E402
    DRIFT_METRICS,
    DUPLICATES_CSV,
    FLAG_BLUR_VAR,
    PROFILE_DIRNAME,
    PROFILE_JSON,
    QUALITY_REPORT,
    STAMP_FILENAME,
    is_profile_current,
    read_baseline,
)


def _checkerboard(size: tuple[int, int] = (32, 32), cell: int = 4) -> Image.Image:
    im = Image.new("L", size, 0)
    px = im.load()
    assert px is not None
    for y in range(size[1]):
        for x in range(size[0]):
            if (x // cell + y // cell) % 2:
                px[x, y] = 255
    return im.convert("RGB")


def _gradient(size: tuple[int, int] = (32, 32)) -> Image.Image:
    """Left-to-right ramp: bright, high-contrast, and a dhash distinct from every other frame."""
    im = Image.new("L", size, 0)
    px = im.load()
    assert px is not None
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = round(x * 255 / (size[0] - 1))
    return im.convert("RGB")


def _write_frame(root: Path, split: str, name: str, image: Image.Image, label_lines: str) -> dict:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    image.save(root / "images" / split / f"{name}.jpg")
    (root / "labels" / split / f"{name}.txt").write_text(label_lines, encoding="utf-8")
    n = len(label_lines.strip().splitlines()) if label_lines.strip() else 0
    return {
        "split": split,
        "sequence": name.rsplit("_", 1)[0],
        "frame_index": int(name.rsplit("_", 1)[-1]),
        "image_relpath": f"images/{split}/{name}.jpg",
        "label_relpath": f"labels/{split}/{name}.txt",
        "width": image.width,
        "height": image.height,
        "n_boxes": n,
        "n_person": n,
        "n_vehicle": 0,
        "n_two_three_wheeler": 0,
    }


@pytest.fixture
def subset(tmp_path: Path) -> Path:
    """Synthetic subset: a duplicate pair in train (solid frames), distinct val/test frames.
    """
    root = tmp_path / "subset"
    # imagehash's dhash sets a bit only where brightness INCREASES left-to-right, so solid
    # frames, constant rows, and monotonically decreasing gradients all hash to zero. A
    # black-left/white-right split has a guaranteed increasing seam -> a distinct, non-zero
    # hash for the test frame.
    half_split = Image.new("L", (32, 32), 0)
    half_split.paste(255, (16, 0, 32, 32))
    half_split = half_split.convert("RGB")
    rows = [
        # Two identical solid frames -> one duplicate cluster (also dark/low-contrast flags).
        _write_frame(
            root,
            "train",
            "seqA_0000001",
            Image.new("RGB", (32, 32), (10, 10, 10)),
            "0 0.5 0.5 0.25 0.25\n",
        ),
        _write_frame(
            root,
            "train",
            "seqA_0000021",
            Image.new("RGB", (32, 32), (10, 10, 10)),
            "0 0.5 0.5 0.25 0.25\n0 0.25 0.25 0.1 0.2\n",
        ),
        _write_frame(root, "train", "seqB_0000001", _gradient(), "0 0.5 0.5 0.25 0.25\n"),
        _write_frame(root, "val", "seqA_0000041", _checkerboard(), "1 0.5 0.5 0.5 0.5\n"),
        _write_frame(root, "test", "seqA_0000061", half_split, "2 0.5 0.5 0.5 0.5\n"),
    ]
    lines = [manifest_header(), *(manifest_csv_line(row) for row in rows)]
    (root / MANIFEST_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


class FakePublisher:
    """Stands in for the data version: records what it was handed, and when."""

    def __init__(self, run_id: str = "data-run-1") -> None:
        self.run_id = run_id
        self.calls: list[tuple[Path, str]] = []
        self.stamp_existed: list[bool] = []

    def __call__(self, profile_dir: Path, stamp: str) -> str:
        self.calls.append((Path(profile_dir), stamp))
        self.stamp_existed.append((Path(profile_dir) / STAMP_FILENAME).exists())
        return self.run_id


def test_stats_combine_fn_matches_statistics_module() -> None:
    fn = StatsCombineFn()
    acc1 = fn.add_input(fn.create_accumulator(), {"x": 1.0, "y": 10.0})
    acc2 = fn.add_input(fn.create_accumulator(), {"x": 3.0, "y": 30.0})
    out = fn.extract_output(fn.merge_accumulators([acc1, acc2]))
    values = [1.0, 3.0]
    assert out["x"]["count"] == 2
    assert out["x"]["mean"] == pytest.approx(statistics.mean(values))
    assert out["x"]["std"] == pytest.approx(statistics.pstdev(values))
    assert (out["x"]["min"], out["x"]["max"]) == (1.0, 3.0)
    assert out["y"]["mean"] == pytest.approx(20.0)


def test_main_end_to_end(subset: Path) -> None:
    assert main([f"--input-dir={subset}", "--runner=DirectRunner"], publish=FakePublisher()) == 0
    profile_dir = subset / PROFILE_DIRNAME

    profile = json.loads((profile_dir / PROFILE_JSON).read_text(encoding="utf-8"))
    assert profile["schema_version"] == 2
    assert set(profile) == {
        "schema_version",
        "source_manifest_sha256",
        "params",
        "dataset",
        "splits",
        "classes",
        "baseline",
    }
    assert profile["dataset"]["n_frames"] == 5
    assert profile["dataset"]["n_boxes"] == 6
    assert profile["splits"]["train"]["n_frames"] == 3
    assert profile["splits"]["train"]["boxes_per_frame"]["mean"] == pytest.approx(4 / 3)
    assert profile["classes"]["train"]["person"]["n_boxes"] == 4
    assert profile["classes"]["val"]["vehicle"]["n_boxes"] == 1
    # The two identical solid frames form exactly one duplicate cluster: counted in the
    # profile, listed row-per-frame in the sibling CSV.
    assert profile["dataset"]["n_duplicate_clusters"] == 1
    with (profile_dir / DUPLICATES_CSV).open(newline="", encoding="utf-8") as fh:
        duplicates = list(csv.DictReader(fh))
    assert len(duplicates) == 2
    assert {r["dhash"] for r in duplicates} == {duplicates[0]["dhash"]}
    assert all(r["image_relpath"].startswith("images/train/seqA_") for r in duplicates)

    # Solid dark frames are flagged (informational) -> report rows exist, run still succeeds.
    with (profile_dir / QUALITY_REPORT).open(newline="", encoding="utf-8") as fh:
        report = list(csv.DictReader(fh))
    assert {r["flag"] for r in report} >= {"dark", "low_contrast"}
    assert all(r["flag"] != "corrupt" for r in report)
    assert profile["dataset"]["n_flagged"] == len(report)

    # Drift baseline: one CSV row per TRAIN sequence carrying that scene's mean of each drift
    # statistic; profile.json keeps only the fixed-size shape of it. The val/test frames also
    # live under the name "seqA" -> seqA's two frames prove the baseline is train-split only.
    assert profile["baseline"] == {"split": "train", "n_sequences": 2}
    scenes = {scene.sequence: scene for scene in read_baseline(profile_dir)}
    assert set(scenes) == {"seqA", "seqB"}
    assert scenes["seqA"].n_frames == 2
    assert scenes["seqB"].n_frames == 1
    seq_a, seq_b = scenes["seqA"].means, scenes["seqB"].means
    assert set(seq_a) == set(DRIFT_METRICS)
    # seqA is two identical solid RGB(10,10,10) frames: flat, so contrast vanishes and the
    # only edge energy is JPEG ringing at the frame border — well under the blurry flag.
    assert seq_a["brightness"] == pytest.approx(10.0, abs=1.0)
    assert seq_a["contrast"] == pytest.approx(0.0, abs=1.0)
    assert seq_a["blur"] < FLAG_BLUR_VAR
    # seqB is a left-to-right ramp: brighter and textured, a visibly different scene.
    assert seq_b["brightness"] > 100.0
    assert seq_b["contrast"] > seq_a["contrast"]
    assert seq_b["blur"] > seq_a["blur"]

    # Stamp lifecycle: current now, stale after the manifest changes.
    assert (profile_dir / STAMP_FILENAME).is_file()
    assert is_profile_current(subset)
    (subset / MANIFEST_FILENAME).write_text("changed", encoding="utf-8")
    assert not is_profile_current(subset)


def test_main_fails_on_corrupt_image_and_withholds_stamp(subset: Path) -> None:
    # Truncate one image: the run must fail (exit 1) and must NOT write the stamp.
    victim = subset / "images" / "train" / "seqA_0000001.jpg"
    victim.write_bytes(victim.read_bytes()[:40])
    publisher = FakePublisher()
    assert main([f"--input-dir={subset}", "--runner=DirectRunner"], publish=publisher) == 1
    profile_dir = subset / PROFILE_DIRNAME
    assert not (profile_dir / STAMP_FILENAME).exists()
    assert not is_profile_current(subset)
    assert publisher.calls == []  # a failed profile is not a data version
    with (profile_dir / QUALITY_REPORT).open(newline="", encoding="utf-8") as fh:
        report = list(csv.DictReader(fh))
    assert any(r["flag"] == "corrupt" for r in report)


def test_the_profile_is_published_before_it_is_stamped(subset: Path) -> None:
    """Stamp-last, extended: the stamp means "profiled *and* published"."""
    publisher = FakePublisher()
    assert main([f"--input-dir={subset}", "--runner=DirectRunner"], publish=publisher) == 0

    profile_dir = subset / PROFILE_DIRNAME
    assert publisher.stamp_existed == [False]  # nothing was current while publishing
    assert (profile_dir / STAMP_FILENAME).is_file()
    assert is_profile_current(subset)

    # The whole directory is what gets published, identified by the very payload stamped after.
    published_dir, stamp = publisher.calls[0]
    assert published_dir == profile_dir
    assert json.loads(stamp) == json.loads(
        (profile_dir / STAMP_FILENAME).read_text(encoding="utf-8")
    )


def test_a_failed_publish_leaves_the_profile_stale(subset: Path) -> None:
    """No silent gap in tracking: the next orchestrated pass re-profiles and retries."""

    def refuse(profile_dir: Path, stamp: str) -> str:
        raise RuntimeError("tracking server unreachable")

    assert main([f"--input-dir={subset}", "--runner=DirectRunner"], publish=refuse) == 1
    assert not (subset / PROFILE_DIRNAME / STAMP_FILENAME).exists()
    assert not is_profile_current(subset)
