"""Auto-label VisDrone-VID sequences: one annotation file per ``sequences/<seq>/`` folder.

``python -m mlops_cv.autolabel <sequence dir>... [--out DIR]`` (docs/autolabel.md).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mlops_cv.startup import StartupError, exits_on_startup_error

logger = logging.getLogger(__name__)


def frame_paths(sequence: Path) -> list[Path]:
    """The sequence's frames in order; ``label_clip`` numbers them by position from 1."""
    frames = sorted(sequence.glob("*.jpg"))
    if not frames:
        raise StartupError(f"no frames in {sequence}: pass a VisDrone-VID sequences/<seq> folder")
    if [f.name for f in frames] != [f"{i:07d}.jpg" for i in range(1, len(frames) + 1)]:
        raise StartupError(f"{sequence}: frames must be 0000001.jpg, 0000002.jpg, … with no gaps")
    return frames


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Auto-label VisDrone-VID sequences.")
    p.add_argument("sequences", nargs="+", type=Path, help="sequences/<seq> folders of frames")
    p.add_argument(
        "--out", type=Path, help="annotation dir (default: <runs_dir>/labeler/annotations)"
    )
    args = p.parse_args(argv)
    clips = {seq: frame_paths(seq) for seq in args.sequences}

    from mlops_cv.autolabel.labeler import Labeler, label_clip, read_rgb
    from mlops_cv.config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    cfg = settings.autolabel
    out_dir = (args.out or cfg.runs_dir / "labeler" / "annotations").resolve()
    labeler = Labeler(cfg.detector, cfg.box_filter, settings.training.device, cfg.max_det)
    out_dir.mkdir(parents=True, exist_ok=True)
    for seq, frames in clips.items():
        lines = label_clip(labeler, cfg.tracker, cfg.confidence_floor, map(read_rgb, frames))
        out = out_dir / f"{seq.name}.txt"
        tmp = out.with_suffix(".part")  # a crash leaves no partial file under the final name
        tmp.write_text("".join(f"{line}\n" for line in lines))
        tmp.replace(out)
        logger.info(f"{seq.name}: {len(frames)} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
