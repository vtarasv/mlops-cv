"""Unit tests for the pure profiling metrics and provenance stamps (pillow/stdlib only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageFilter
from pydantic import ValidationError

from mlops_cv.data.convert_visdrone_vid import YoloBox
from mlops_cv.pipelines import provenance
from mlops_cv.pipelines.profiling import (
    BASELINE_FIELDS,
    DRIFT_METRICS,
    PROFILE_DIRNAME,
    PROFILE_PARAMS,
    STAMP_FILENAME,
    BaselineScene,
    baseline_csv_line,
    baseline_header,
    baseline_rows,
    blur_score,
    box_stats,
    brightness_contrast,
    dhash,
    frame_flags,
    is_profile_current,
    profile_image_bytes,
)


def _checkerboard(size: int = 64, cell: int = 8) -> Image.Image:
    im = Image.new("L", (size, size), 0)
    px = im.load()
    assert px is not None
    for y in range(size):
        for x in range(size):
            if (x // cell + y // cell) % 2:
                px[x, y] = 255
    return im.convert("RGB")


def _jpeg_bytes(im: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()


# --- metrics ---


def test_brightness_contrast_solid_image() -> None:
    mean, std = brightness_contrast(Image.new("RGB", (32, 32), (128, 128, 128)))
    assert mean == pytest.approx(128.0)
    assert std == pytest.approx(0.0)


def test_brightness_contrast_checkerboard_high_contrast() -> None:
    mean, std = brightness_contrast(_checkerboard())
    assert mean == pytest.approx(127.5, abs=1.0)
    assert std > 100.0  # black/white halves -> near-maximal stddev


def test_blur_score_orders_sharp_above_blurred() -> None:
    sharp = _checkerboard()
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=4))
    assert blur_score(sharp) > blur_score(blurred)


def test_dhash_identical_for_identical_images() -> None:
    a, b = _checkerboard(), _checkerboard()
    assert dhash(a) == dhash(b)
    assert len(dhash(a)) == 16  # 64 bits as hex


def test_dhash_differs_for_different_content() -> None:
    gradient = Image.linear_gradient("L").resize((64, 64)).convert("RGB")
    assert dhash(_checkerboard()) != dhash(gradient)


def test_profile_image_bytes_returns_metrics() -> None:
    out = profile_image_bytes(_jpeg_bytes(_checkerboard()))
    assert set(out) == {"width", "height", "brightness", "contrast", "blur", "dhash"}
    assert (out["width"], out["height"]) == (64, 64)


def test_profile_image_bytes_raises_on_corrupt() -> None:
    truncated = _jpeg_bytes(_checkerboard())[:40]
    with pytest.raises(Exception):  # noqa: B017 - any decode failure counts as corrupt
        profile_image_bytes(truncated)


def test_box_stats_area_and_aspect() -> None:
    stats = box_stats([YoloBox(0, 0.5, 0.5, 0.2, 0.1), YoloBox(1, 0.5, 0.5, 0.1, 0.0)])
    assert stats[0]["area"] == pytest.approx(0.02)
    assert stats[0]["aspect"] == pytest.approx(2.0)
    assert stats[1]["aspect"] == 0.0  # zero-height guard


@pytest.mark.parametrize(
    "metrics,expected",
    [
        ({"brightness": 10.0, "contrast": 50.0, "blur": 500.0}, ["dark"]),
        ({"brightness": 240.0, "contrast": 50.0, "blur": 500.0}, ["bright"]),
        ({"brightness": 128.0, "contrast": 5.0, "blur": 500.0}, ["low_contrast"]),
        ({"brightness": 128.0, "contrast": 50.0, "blur": 10.0}, ["blurry"]),
        ({"brightness": 128.0, "contrast": 50.0, "blur": 500.0}, []),
        ({"brightness": 10.0, "contrast": 5.0, "blur": 10.0}, ["dark", "low_contrast", "blurry"]),
    ],
)
def test_frame_flags(metrics: dict, expected: list[str]) -> None:
    assert frame_flags(metrics) == expected


# --- drift baseline CSV contract ---


def test_baseline_fields_are_the_pinned_column_order() -> None:
    """The column order is a cross-process contract: the monitor reads what profiling wrote."""
    assert BASELINE_FIELDS == ["sequence", "n_frames", "brightness", "contrast", "blur"]
    assert DRIFT_METRICS == ("brightness", "contrast", "blur")
    assert baseline_header() == "sequence,n_frames,brightness,contrast,blur"


def test_baseline_csv_line_round_trips_with_typed_values() -> None:
    scene = BaselineScene(
        sequence="uav0000013_00000_v",
        n_frames=14,
        means={"brightness": 96.5553410021416, "contrast": 39.26017475352093, "blur": 963.157193},
    )
    assert baseline_rows([baseline_header(), baseline_csv_line(scene)]) == [scene]


def test_baseline_rows_rejects_a_foreign_header() -> None:
    with pytest.raises(ValueError, match="unexpected drift baseline header"):
        baseline_rows(["sequence,brightness", "seqA,1.0"])


def test_baseline_scene_rejects_means_that_are_not_the_drift_metrics() -> None:
    with pytest.raises(ValidationError, match="must be keyed by"):
        BaselineScene(sequence="seqA", n_frames=1, means={"brightness": 1.0})


# --- provenance stamps ---


def test_stamp_payload_is_canonical() -> None:
    a = provenance.stamp_payload("sha", {"b": 1, "a": 2})
    b = provenance.stamp_payload("sha", {"a": 2, "b": 1})
    assert a == b
    assert json.loads(a) == {"source_manifest_sha256": "sha", "params": {"a": 2, "b": 1}}


@pytest.fixture
def subset(tmp_path: Path) -> Path:
    root = tmp_path / "subset"
    (root / PROFILE_DIRNAME).mkdir(parents=True)
    (root / "manifest.csv").write_text("split,sequence\ntrain,seq1\n", encoding="utf-8")
    return root


def _write_stamp(root: Path, params: dict) -> None:
    payload = provenance.stamp_payload(provenance.manifest_sha256(root / "manifest.csv"), params)
    (root / PROFILE_DIRNAME / STAMP_FILENAME).write_text(payload, encoding="utf-8")


def test_is_profile_current_true_on_matching_stamp(subset: Path) -> None:
    _write_stamp(subset, PROFILE_PARAMS)
    assert is_profile_current(subset)


def test_is_profile_current_false_without_stamp(subset: Path) -> None:
    assert not is_profile_current(subset)


def test_is_profile_current_false_after_manifest_change(subset: Path) -> None:
    _write_stamp(subset, PROFILE_PARAMS)
    (subset / "manifest.csv").write_text("split,sequence\ntrain,seq2\n", encoding="utf-8")
    assert not is_profile_current(subset)


def test_is_profile_current_false_on_pre_baseline_stamp(subset: Path) -> None:
    """A profile stamped before the drift baseline existed reads stale, so the DAG re-profiles."""
    legacy = {k: v for k, v in PROFILE_PARAMS.items() if not k.startswith("baseline_")}
    assert legacy != PROFILE_PARAMS, "the baseline parameters must be part of the profile stamp"
    _write_stamp(subset, legacy)
    assert not is_profile_current(subset)


def test_is_profile_current_false_on_param_drift(subset: Path) -> None:
    _write_stamp(subset, {**PROFILE_PARAMS, "blur_var": 999.0})
    assert not is_profile_current(subset)


def test_is_current_false_on_corrupt_stamp(subset: Path) -> None:
    (subset / PROFILE_DIRNAME / STAMP_FILENAME).write_text("not json", encoding="utf-8")
    assert not is_profile_current(subset)


def test_is_current_false_when_manifest_missing(tmp_path: Path) -> None:
    assert not provenance.is_current(tmp_path / "nope.csv", tmp_path / "stamp.json", {})
