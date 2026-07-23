"""Unit tests for the demo visualizer's box drawing + clip-config parsing (PIL only, no GPU)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from mlops_cv.data.convert_visdrone_vid import YoloBox
from mlops_cv.eval.visualize import (
    ClipSpec,
    DrawBox,
    draw_boxes,
    load_demo_clips,
    yolo_to_pixel,
)


def test_yolo_to_pixel_center_to_corners() -> None:
    px = yolo_to_pixel(YoloBox(cls=0, xc=0.5, yc=0.5, w=0.2, h=0.4), 100, 200)
    assert (px.x1, px.y1, px.x2, px.y2) == (40.0, 60.0, 60.0, 140.0)
    assert px.label == "person"


def test_draw_boxes_in_place_marks_pixels() -> None:
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    out = draw_boxes(img, [DrawBox(10, 10, 40, 40, "person")], (0, 200, 0))
    assert out is img  # drawn in place, same object returned
    assert img.getbbox() is not None  # something non-black was drawn


def test_draw_boxes_empty_leaves_image_blank() -> None:
    img = Image.new("RGB", (16, 16), (0, 0, 0))
    draw_boxes(img, [], (220, 40, 40), with_conf=True)
    assert img.getbbox() is None


def test_load_demo_clips_parses_strings_and_mappings(tmp_path: Path) -> None:
    cfg_path = tmp_path / "demo.yaml"
    cfg_path.write_text(
        "split: val\nfps: 25\nmax_side: 960\n"
        "clips:\n  - seqA\n  - {sequence: seqB, start: 5, length: 10}\n",
        encoding="utf-8",
    )
    cfg = load_demo_clips(cfg_path)
    assert (cfg.split, cfg.fps, cfg.max_side) == ("val", 25, 960)
    assert cfg.clips == [ClipSpec("seqA", 1, None), ClipSpec("seqB", 5, 10)]


def test_committed_demo_clips_config_is_full_length() -> None:
    cfg = load_demo_clips("configs/demo_clips.yaml")
    assert cfg.split == "test"
    assert [c.sequence for c in cfg.clips] == [
        "uav0000161_00000_v",
        "uav0000355_00001_v",
        "uav0000073_00600_v",
    ]
    assert all(c.start == 1 and c.length is None for c in cfg.clips)  # full length, no trim
