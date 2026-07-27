"""Beam pipeline: build the YOLO training subset (and demo store) from raw VisDrone-VID.

Keeps **every** train/val/test sequence (max scene diversity) and shrinks the set by
``--frame-stride`` (keep every Nth 1-based frame), which removes near-duplicate consecutive
video frames. Kept frames land as flattened ``images|labels/{split}/<seq>_<NNNNNNN>.{jpg,txt}``
plus a ``manifest.csv`` and an absolute-path copy of the dataset YAML. Only frames with at least
one surviving box are materialized.

The pipeline also materializes the **demo store**: for the clips in ``configs/demo_clips.yaml``
it copies ALL frames (full frame rate, empties included) to ``demo/images/<seq>/`` with
per-frame YOLO ground-truth labels under ``demo/labels/<seq>/`` — evaluation renders demo videos
from here, so the raw dataset is needed only by this pipeline.

Run locally:
    uv run python -m mlops_cv.pipelines.ingest_pipeline --runner DirectRunner
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from pathlib import Path

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.io.textio import WriteToText
from apache_beam.metrics.metric import Metrics, MetricsFilter
from apache_beam.options.pipeline_options import PipelineOptions
from PIL import Image

from mlops_cv.config import get_settings
from mlops_cv.data.convert_visdrone_vid import convert_sequence_text, labels_to_text
from mlops_cv.data.subset import (
    DEMO_DIRNAME,
    IMAGES_DIRNAME,
    LABELS_DIRNAME,
    MANIFEST_FIELDS,
    MANIFEST_FILENAME,
    SPLITS,
    demo_image_relpath,
    demo_label_relpath,
    frame_name,
    image_relpath,
    label_relpath,
    manifest_csv_line,
    manifest_header,
    raw_split_dir,
    stamp_dataset_yaml,
)
from mlops_cv.eval.visualize import ClipSpec, load_demo_clips

logger = logging.getLogger(__name__)

COUNTER_NAMESPACE = "ingest"


class IngestOptions(PipelineOptions):
    """Pipeline options; ``--runner`` etc. come from the Beam standard options."""

    @classmethod
    def _add_argparse_args(cls, parser) -> None:  # noqa: ANN001 - Beam's argparse hook
        parser.add_argument(
            "--raw-dir",
            default=None,
            help="raw VisDrone-VID dir holding VisDrone2019-VID-{train,val,test-dev}/ "
            "(default: settings.data.raw_dir)",
        )
        parser.add_argument(
            "--output-dir", default=None, help="subset dir (default: settings.data.subset_dir)"
        )
        parser.add_argument(
            "--frame-stride", type=int, default=20, help="keep every Nth frame (all splits)"
        )
        parser.add_argument(
            "--sequences", nargs="+", default=None, help="restrict to these sequences"
        )
        parser.add_argument(
            "--template-yaml",
            default=None,
            help="dataset YAML template to stamp (default: settings.data.dataset_yaml)",
        )
        parser.add_argument(
            "--demo-clips",
            default=None,
            help="demo-clips YAML for the demo store (default: configs/demo_clips.yaml; "
            "'none' disables)",
        )


def _image_size_from_bytes(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as im:
        return im.size


def _match_frames(sequence_dir: str) -> dict[int, str]:
    """``{frame_index: path}`` for a sequence's JPEG frames via portable matching."""
    result = FileSystems.match([FileSystems.join(sequence_dir, "*.jpg")])[0]
    return {int(Path(m.path).stem): m.path for m in result.metadata_list}


def _read_text(path: str) -> str:
    with FileSystems.open(path) as fh:
        return fh.read().decode("utf-8")


class ConvertSequenceDoFn(beam.DoFn):
    """Convert one ``(split, sequence)`` into per-frame records for the strided subset."""

    def __init__(self, raw_dir: str, stride: int) -> None:
        super().__init__()
        self.raw_dir = raw_dir
        self.stride = stride
        self.sequences_processed = Metrics.counter(COUNTER_NAMESPACE, "sequences_processed")
        self.sequences_empty = Metrics.counter(COUNTER_NAMESPACE, "sequences_empty")
        self.frames_kept = Metrics.counter(COUNTER_NAMESPACE, "frames_kept")

    def process(self, element: tuple[str, str]) -> Iterator[dict]:
        split, seq = element
        self.sequences_processed.inc()
        base = FileSystems.join(self.raw_dir, raw_split_dir(split))
        frames = _match_frames(FileSystems.join(base, "sequences", seq))
        if not frames:
            self.sequences_empty.inc()
            return
        # Frame size is constant within a VisDrone-VID sequence.
        with FileSystems.open(frames[min(frames)]) as fh:
            width, height = _image_size_from_bytes(fh.read())
        text = _read_text(FileSystems.join(base, "annotations", f"{seq}.txt"))
        by_frame = convert_sequence_text(text, (width, height))
        for idx in sorted(by_frame):  # only frames that have surviving boxes
            if (idx - 1) % self.stride != 0 or idx not in frames:
                continue
            name = frame_name(seq, idx)
            boxes = by_frame[idx]
            counts = {0: 0, 1: 0, 2: 0}
            for box in boxes:
                counts[box.cls] += 1
            self.frames_kept.inc()
            yield {
                "split": split,
                "sequence": seq,
                "frame_index": idx,
                "image_relpath": image_relpath(split, name),
                "label_relpath": label_relpath(split, name),
                "width": width,
                "height": height,
                "n_boxes": len(boxes),
                "n_person": counts[0],
                "n_vehicle": counts[1],
                "n_two_three_wheeler": counts[2],
                "src_image": frames[idx],
                "label_text": labels_to_text(boxes),
            }


class MaterializeFrameDoFn(beam.DoFn):
    """Copy one frame + write its label file; emits the manifest row."""

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.images_copied = Metrics.counter(COUNTER_NAMESPACE, "images_copied")

    def process(self, item: dict) -> Iterator[dict]:
        FileSystems.copy(
            [item["src_image"]], [FileSystems.join(self.output_dir, item["image_relpath"])]
        )
        with FileSystems.create(FileSystems.join(self.output_dir, item["label_relpath"])) as fh:
            fh.write(item["label_text"].encode("utf-8"))
        self.images_copied.inc()
        yield {field: item[field] for field in MANIFEST_FIELDS}


class ExpandDemoClipDoFn(beam.DoFn):
    """Expand one demo clip into per-frame copy tasks (full frame rate, empties included)."""

    def __init__(self, raw_dir: str, split: str) -> None:
        super().__init__()
        self.raw_dir = raw_dir
        self.split = split  # YOLO split name, e.g. "test"
        self.demo_clips = Metrics.counter(COUNTER_NAMESPACE, "demo_clips")

    def process(self, clip: ClipSpec) -> Iterator[dict]:
        self.demo_clips.inc()
        base = FileSystems.join(self.raw_dir, raw_split_dir(self.split))
        frames = _match_frames(FileSystems.join(base, "sequences", clip.sequence))
        if not frames:
            logger.warning(f"demo clip {clip.sequence} has no frames under {base}")
            return
        with FileSystems.open(frames[min(frames)]) as fh:
            width, height = _image_size_from_bytes(fh.read())
        text = _read_text(FileSystems.join(base, "annotations", f"{clip.sequence}.txt"))
        by_frame = convert_sequence_text(text, (width, height))
        end = None if clip.length is None else clip.start + clip.length
        for idx in sorted(frames):
            if idx < clip.start or (end is not None and idx >= end):
                continue
            boxes = by_frame.get(idx, [])
            yield {
                "sequence": clip.sequence,
                "src_image": frames[idx],
                "dest_image": demo_image_relpath(clip.sequence, idx),
                # No boxes -> no label file: the renderer treats a missing file as "no GT".
                "label_text": labels_to_text(boxes) if boxes else None,
                "dest_label": demo_label_relpath(clip.sequence, idx),
            }


class MaterializeDemoFrameDoFn(beam.DoFn):
    """Copy one demo frame + write its GT label file (when the frame has boxes)."""

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.demo_frames_copied = Metrics.counter(COUNTER_NAMESPACE, "demo_frames_copied")

    def process(self, item: dict) -> None:
        FileSystems.copy(
            [item["src_image"]], [FileSystems.join(self.output_dir, item["dest_image"])]
        )
        if item["label_text"] is not None:
            with FileSystems.create(FileSystems.join(self.output_dir, item["dest_label"])) as fh:
                fh.write(item["label_text"].encode("utf-8"))
        self.demo_frames_copied.inc()


def _list_sequences(raw_dir: str, split: str) -> list[str]:
    """Sorted sequence names of one raw split, from its annotation files (portable matching)."""
    pattern = FileSystems.join(raw_dir, raw_split_dir(split), "annotations", "*.txt")
    names = sorted(Path(m.path).stem for m in FileSystems.match([pattern])[0].metadata_list)
    if not names:
        raise RuntimeError(
            f"VisDrone-VID {split} annotations not found under {pattern}. Download the dataset "
            "(see docs/data.md) and point --raw-dir/DATA__RAW_DIR at the directory holding "
            "VisDrone2019-VID-{train,val,test-dev}/."
        )
    return names


def _prepare_output_dirs(output_dir: str, demo_sequences: list[str]) -> None:
    """Clean rebuild: drop stale outputs, then pre-create every dir the workers write into.

    ``FileSystems.copy`` does not create parent directories (and ``mkdirs`` raises on an
    existing leaf), so everything is created here up front.
    """
    for sub in (IMAGES_DIRNAME, LABELS_DIRNAME, DEMO_DIRNAME):
        root = FileSystems.join(output_dir, sub)
        if FileSystems.exists(root):
            FileSystems.delete([root])
    for sub in (IMAGES_DIRNAME, LABELS_DIRNAME):
        for split in SPLITS:
            FileSystems.mkdirs(FileSystems.join(output_dir, sub, split))
        for seq in demo_sequences:
            FileSystems.mkdirs(FileSystems.join(output_dir, DEMO_DIRNAME, sub, seq))


def _query_counters(result) -> dict[str, int]:  # noqa: ANN001 - Beam PipelineResult
    metrics = result.metrics().query(MetricsFilter().with_namespace(COUNTER_NAMESPACE))
    return {m.key.metric.name: m.result for m in metrics["counters"]}


def run(options: IngestOptions) -> int:
    """Execute the ingestion pipeline; assumes all options are fully resolved (see ``main``)."""
    raw_dir, output_dir = options.raw_dir, options.output_dir

    elements = [
        (split, seq)
        for split in SPLITS
        for seq in _list_sequences(raw_dir, split)
        if options.sequences is None or seq in options.sequences
    ]
    if options.sequences:
        matched = {seq for _, seq in elements}
        if unknown := [s for s in options.sequences if s not in matched]:
            raise ValueError(f"--sequences not found in any split: {unknown}")

    demo_config = None
    if options.demo_clips != "none":
        demo_config = load_demo_clips(options.demo_clips)
    demo_sequences = [c.sequence for c in demo_config.clips] if demo_config else []
    _prepare_output_dirs(output_dir, demo_sequences)

    with beam.Pipeline(options=options) as p:
        _ = (
            p
            | "CreateSequences" >> beam.Create(elements)
            | "ConvertSequences" >> beam.ParDo(ConvertSequenceDoFn(raw_dir, options.frame_stride))
            # Fusion break: sequences can be heavily skewed; rebalance the per-frame IO.
            | "RebalanceFrames" >> beam.Reshuffle()
            | "MaterializeFrames" >> beam.ParDo(MaterializeFrameDoFn(output_dir))
            | "ToCsvLine" >> beam.Map(manifest_csv_line)
            | "WriteManifest"
            >> WriteToText(
                FileSystems.join(output_dir, MANIFEST_FILENAME),
                shard_name_template="",
                header=manifest_header(),
            )
        )
        if demo_config:
            _ = (
                p
                | "CreateDemoClips" >> beam.Create(demo_config.clips)
                | "ExpandDemoClips" >> beam.ParDo(ExpandDemoClipDoFn(raw_dir, demo_config.split))
                | "RebalanceDemoFrames" >> beam.Reshuffle()
                | "MaterializeDemoFrames" >> beam.ParDo(MaterializeDemoFrameDoFn(output_dir))
            )

    counters = _query_counters(p.result)
    kept, copied = counters.get("frames_kept", 0), counters.get("images_copied", 0)
    if kept != copied:
        logger.warning(f"frame count mismatch: {kept} converted vs {copied} copied")
    yaml_path = stamp_dataset_yaml(Path(options.template_yaml), Path(output_dir))
    logger.info(
        f"ingested {counters.get('sequences_processed', 0)} sequences -> {copied} frames "
        f"| demo store: {counters.get('demo_frames_copied', 0)} frames "
        f"across {len(demo_sequences)} clips | wrote {yaml_path.name}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    options = IngestOptions(argv)
    if options.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    # Settings-backed defaults, resolved at run time (never at class definition).
    options.raw_dir = options.raw_dir or str(settings.data.raw_dir)
    options.output_dir = options.output_dir or str(settings.data.subset_dir)
    options.template_yaml = options.template_yaml or str(settings.data.dataset_yaml)
    options.demo_clips = options.demo_clips or "configs/demo_clips.yaml"
    return run(options)


if __name__ == "__main__":
    raise SystemExit(main())
