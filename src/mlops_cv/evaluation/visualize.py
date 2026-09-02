"""Render qualitative ground-truth-vs-prediction demo videos.

For a fixed set of held-out clips (``configs/demo_clips.yaml``), overlay the
**ground-truth** boxes (one colour) and the model's **predictions** (another colour, with
confidence) on each frame and encode an ``.mp4`` per clip. The same clips every run keep
experiments visually comparable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

from mlops_cv.data.convert_visdrone_vid import YOLO_NAMES, YoloBox, image_size, read_yolo_labels
from mlops_cv.data.subset import IMAGES_DIRNAME, LABELS_DIRNAME

GT_COLOR = (0, 200, 0)  # ground truth: green
PRED_COLOR = (220, 40, 40)  # predictions: red


@dataclass(frozen=True)
class ClipSpec:
    """One demo clip: a sequence plus an optional ``[start, start+length)`` frame window."""

    sequence: str
    start: int = 1
    length: int | None = None  # None -> render to the end of the sequence


@dataclass(frozen=True)
class DemoClipsConfig:
    """Parsed ``demo_clips.yaml``: source split, playback fps, downscale cap, and the clips."""

    # YOLO split name, e.g. "test". Used only by the ingest pipeline (it materializes the
    # clips into the subset's demo store); rendering never touches raw data.
    split: str
    fps: int
    max_side: int
    clips: list[ClipSpec]


@dataclass(frozen=True)
class DrawBox:
    """A pixel-space box to draw, with an optional label and confidence."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""
    conf: float | None = None


def _parse_clip(item: object) -> ClipSpec:
    if isinstance(item, str):
        return ClipSpec(sequence=item)
    if isinstance(item, dict) and "sequence" in item:
        return ClipSpec(
            sequence=str(item["sequence"]),
            start=int(item.get("start", 1)),
            length=None if item.get("length") is None else int(item["length"]),
        )
    raise ValueError(f"invalid clip entry: {item!r} (expected a sequence name or a mapping)")


def load_demo_clips(path: str | Path) -> DemoClipsConfig:
    """Parse the demo-clips YAML into a :class:`DemoClipsConfig`."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return DemoClipsConfig(
        split=str(data.get("split", "test")),
        fps=int(data.get("fps", 30)),
        max_side=int(data.get("max_side", 1280)),
        clips=[_parse_clip(c) for c in data["clips"]],
    )


def yolo_to_pixel(box: YoloBox, width: int, height: int) -> DrawBox:
    """Convert a normalized center-form :class:`YoloBox` to a pixel-space corner-form box."""
    half_w, half_h = box.w / 2, box.h / 2
    return DrawBox(
        x1=(box.xc - half_w) * width,
        y1=(box.yc - half_h) * height,
        x2=(box.xc + half_w) * width,
        y2=(box.yc + half_h) * height,
        label=YOLO_NAMES.get(box.cls, str(box.cls)),
    )


def _line_width(height: int) -> int:
    """Box-outline thickness scaled to the frame height (min 2 px)."""
    return max(2, round(height / 360))


def _label_font_size(height: int) -> int:
    """Compact label font size scaled to the frame height (min 12 px)."""
    return max(12, round(height / 70))


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Pillow's bundled scalable, antialiased font at ``size`` px (no external font file needed)."""
    return ImageFont.load_default(size=size)  # type: ignore


def draw_boxes(
    image: Image.Image,
    boxes: Iterable[DrawBox],
    color: tuple[int, int, int],
    *,
    width: int = 2,
    font: ImageFont.FreeTypeFont | None = None,
    with_conf: bool = False,
) -> Image.Image:
    """Draw ``boxes`` onto ``image`` in ``color`` (in place); return the same image.

    Each label is compact antialiased text in ``color`` with a thin dark outline — no filled
    background, so it never occludes neighbouring boxes — placed above the box (or just inside the
    top edge when there is no room). ``font`` defaults to a scalable font sized to the image height;
    pass one in to render many frames with a single shared font.
    """
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=width)
        label = box.label
        if with_conf and box.conf is not None:
            label = f"{label} {box.conf:.2f}".strip()
        if not label:
            continue
        if font is None:
            font = _load_font(_label_font_size(image.height))
        stroke = max(1, round(font.size / 16))
        _, top, _, bottom = font.getbbox(label)
        label_y = box.y1 - (bottom - top) - 2 * stroke
        if label_y < 0:  # no room above the box — sit the label just inside the top edge
            label_y = box.y1 + stroke
        draw.text(
            (box.x1 + stroke, label_y),
            label,
            font=font,
            fill=color,
            stroke_width=stroke,
            stroke_fill=(0, 0, 0),
            anchor="lt",
        )
    return image


def _frame_window(frame_paths: list[Path], clip: ClipSpec) -> list[Path]:
    """The frames of ``clip`` in order: those with frame index in ``[start, start+length)``."""
    end = None if clip.length is None else clip.start + clip.length
    return [
        p for p in frame_paths if clip.start <= int(p.stem) and (end is None or int(p.stem) < end)
    ]


def render_demo_clips(
    model: object,
    config: DemoClipsConfig,
    demo_dir: str | Path,
    out_dir: str | Path,
) -> list[Path]:
    """Render one GT-vs-prediction ``.mp4`` per clip; return the written paths.

    Frames and per-frame YOLO ground-truth labels come from the subset's **demo store**
    (``<subset>/demo/images|labels/<sequence>/``.
    """
    import imageio.v2 as imageio
    import numpy as np
    from ultralytics import YOLO

    if not hasattr(model, "predict"):
        model = YOLO(str(model))
    names = getattr(model, "names", YOLO_NAMES)

    demo_dir = Path(demo_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for clip in config.clips:
        frame_paths = sorted((demo_dir / IMAGES_DIRNAME / clip.sequence).glob("*.jpg"))
        frames = _frame_window(frame_paths, clip)
        if not frames:
            continue
        src_w, src_h = image_size(frames[0])
        scale = min(1.0, config.max_side / max(src_w, src_h))
        dst_w, dst_h = round(src_w * scale), round(src_h * scale)
        dst_w, dst_h = dst_w - dst_w % 2, dst_h - dst_h % 2  # even dims required by yuv420p/libx264
        line_w = _line_width(dst_h)
        font = _load_font(_label_font_size(dst_h))
        labels_dir = demo_dir / LABELS_DIRNAME / clip.sequence

        dest = out_dir / f"{clip.sequence}.mp4"
        writer = imageio.get_writer(
            str(dest),
            fps=config.fps,
            codec="libx264",
            macro_block_size=1,  # keep exact (even) dims rather than padding to a multiple of 16
            pixelformat="yuv420p",
            output_params=["-crf", "23"],
        )
        for fp in frames:
            frame = Image.open(fp).convert("RGB").resize((dst_w, dst_h))
            result = model.predict(frame, verbose=False)[0]  # type: ignore
            assert result.boxes is not None, "model.predict() returned no boxes"
            preds = [
                DrawBox(*[float(v) for v in xyxy], label=names[int(cls)], conf=float(conf))
                for xyxy, cls, conf in zip(
                    result.boxes.xyxy.tolist(),
                    result.boxes.cls.tolist(),
                    result.boxes.conf.tolist(),
                    strict=True,
                )
            ]
            # Missing label file == no GT on that frame (the demo store only writes non-empty).
            gt_boxes = read_yolo_labels(labels_dir / f"{fp.stem}.txt")
            gt = [yolo_to_pixel(b, dst_w, dst_h) for b in gt_boxes]
            draw_boxes(frame, gt, GT_COLOR, width=line_w, font=font)
            draw_boxes(frame, preds, PRED_COLOR, width=line_w, font=font, with_conf=True)
            writer.append_data(np.asarray(frame))

        writer.close()
        written.append(dest)
    return written
