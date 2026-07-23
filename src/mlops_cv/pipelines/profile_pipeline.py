"""Beam pipeline: profile a YOLO dataset — per-frame quality metrics + drift baseline.

Maps the pure metrics from :mod:`mlops_cv.pipelines.profiling` over every manifest frame, then
aggregates dataset-level statistics (``CombinePerKey`` per split and per class,
``GroupByKey`` duplicate clustering on perceptual hashes) into two artifacts under
``<subset>/profile/``:

- ``profile.json`` — the dataset profile / **drift baseline** a monitoring component compares
  future data windows against;
- ``quality_report.csv`` — one row per flagged frame (corrupt / dark / bright / low_contrast /
  blurry) for human review.

Flag thresholds are informational; the run **fails only when corrupt/unreadable images exist**
(exit 1, and the provenance stamp is withheld so the next orchestrated run re-profiles).

Run locally:
    uv run python -m mlops_cv.pipelines.profile_pipeline --runner DirectRunner
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
from collections.abc import Iterator
from pathlib import Path

import apache_beam as beam
from apache_beam import pvalue
from apache_beam.io.filesystems import FileSystems
from apache_beam.io.textio import WriteToText
from apache_beam.metrics.metric import Metrics, MetricsFilter
from apache_beam.options.pipeline_options import PipelineOptions

from mlops_cv.config import get_settings
from mlops_cv.data.convert_visdrone_vid import YOLO_NAMES, labels_from_text
from mlops_cv.data.manifest import SPLITS
from mlops_cv.pipelines import profiling, provenance

logger = logging.getLogger(__name__)

COUNTER_NAMESPACE = "profile"
REPORT_FIELDS = ["split", "sequence", "frame_index", "image_relpath", "flag", "detail"]
_FLAG_METRIC = {
    "dark": "brightness",
    "bright": "brightness",
    "low_contrast": "contrast",
    "blurry": "blur",
}


class ProfileOptions(PipelineOptions):
    """Pipeline options; ``--runner`` etc. come from the Beam standard options."""

    @classmethod
    def _add_argparse_args(cls, parser) -> None:  # noqa: ANN001 - Beam's argparse hook
        parser.add_argument(
            "--input-dir",
            default=None,
            help="dataset dir with images/, labels/, manifest.csv "
            "(default: settings.data.subset_dir)",
        )


class ProfileFrameDoFn(beam.DoFn):
    """Profile one manifest row's frame; corrupt images go to a tagged side output."""

    def __init__(self, input_dir: str) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.frames_profiled = Metrics.counter(COUNTER_NAMESPACE, "frames_profiled")
        self.corrupt_images = Metrics.counter(COUNTER_NAMESPACE, "corrupt_images")

    def process(self, row: dict[str, str]) -> Iterator[object]:
        base = {
            "split": row["split"],
            "sequence": row["sequence"],
            "frame_index": row["frame_index"],
            "image_relpath": row["image_relpath"],
        }
        try:
            with FileSystems.open(FileSystems.join(self.input_dir, row["image_relpath"])) as fh:
                metrics = profiling.profile_image_bytes(fh.read())
        except Exception as exc:  # noqa: BLE001 - any decode failure counts as corrupt
            self.corrupt_images.inc()
            yield pvalue.TaggedOutput("corrupt", {**base, "flag": "corrupt", "detail": str(exc)})
            return
        with FileSystems.open(FileSystems.join(self.input_dir, row["label_relpath"])) as fh:
            boxes = labels_from_text(fh.read().decode("utf-8"))
        self.frames_profiled.inc()
        for box, stats in zip(boxes, profiling.box_stats(boxes), strict=True):
            yield pvalue.TaggedOutput("boxes", ((row["split"], YOLO_NAMES[box.cls]), stats))
        yield {**base, **metrics, "n_boxes": len(boxes)}


class StatsCombineFn(beam.CombineFn):
    """Aggregate ``{metric: float}`` dicts into ``{metric: {count, mean, std, min, max}}``."""

    def create_accumulator(self) -> dict[str, tuple[int, float, float, float, float]]:
        return {}

    def add_input(self, acc: dict, element: dict[str, float]) -> dict:
        for name, value in element.items():
            n, total, sumsq, lo, hi = acc.get(name, (0, 0.0, 0.0, math.inf, -math.inf))
            acc[name] = (
                n + 1,
                total + value,
                sumsq + value * value,
                min(lo, value),
                max(hi, value),
            )
        return acc

    def merge_accumulators(self, accumulators) -> dict:  # noqa: ANN001 - Beam API
        merged: dict[str, tuple[int, float, float, float, float]] = {}
        for acc in accumulators:
            for name, (n, total, sumsq, lo, hi) in acc.items():
                mn, mtotal, msumsq, mlo, mhi = merged.get(name, (0, 0.0, 0.0, math.inf, -math.inf))
                merged[name] = (mn + n, mtotal + total, msumsq + sumsq, min(mlo, lo), max(mhi, hi))
        return merged

    def extract_output(self, acc: dict) -> dict[str, dict[str, float]]:
        out = {}
        for name, (n, total, sumsq, lo, hi) in acc.items():
            mean = total / n if n else 0.0
            out[name] = {
                "count": n,
                "mean": mean,
                "std": math.sqrt(max(0.0, sumsq / n - mean * mean)) if n else 0.0,
                "min": lo if n else 0.0,
                "max": hi if n else 0.0,
            }
        return out


def _split_metrics(record: dict) -> tuple[str, dict[str, float]]:
    return record["split"], {
        "brightness": record["brightness"],
        "contrast": record["contrast"],
        "blur": record["blur"],
        "boxes_per_frame": float(record["n_boxes"]),
    }


def _flag_rows(record: dict) -> Iterator[dict]:
    for flag in profiling.frame_flags(record):
        yield {
            "split": record["split"],
            "sequence": record["sequence"],
            "frame_index": record["frame_index"],
            "image_relpath": record["image_relpath"],
            "flag": flag,
            "detail": f"{_FLAG_METRIC[flag]}={record[_FLAG_METRIC[flag]]:.2f}",
        }


def _duplicate_clusters(element: tuple[str, list[str]]) -> Iterator[dict]:
    dhash, relpaths = element
    if len(relpaths) > 1:
        yield {"dhash": dhash, "frames": sorted(relpaths)}


def _report_csv_line(row: dict) -> str:
    buf = io.StringIO()
    csv.writer(buf).writerow([row[field] for field in REPORT_FIELDS])
    return buf.getvalue().rstrip("\r\n")


def _build_profile(
    _,  # noqa: ANN001 - the single trigger element
    split_stats: dict,
    class_stats: list,
    duplicates: list,
    report_rows: list,
    manifest_sha: str,
    output_path: str,
) -> None:
    """Assemble the profile from the aggregated side inputs and write it as one JSON file."""
    classes: dict[str, dict] = {}
    n_boxes = 0
    for (split, class_name), stats in class_stats:
        n_boxes += stats["area"]["count"]
        classes.setdefault(split, {})[class_name] = {
            "n_boxes": stats["area"]["count"],
            "box_area": stats["area"],
            "box_aspect": stats["aspect"],
        }
    splits = {
        split: {"n_frames": stats["brightness"]["count"], **stats}
        for split, stats in split_stats.items()
    }
    profile = {
        "schema_version": 1,
        "source_manifest_sha256": manifest_sha,
        "params": profiling.PROFILE_PARAMS,
        "dataset": {
            "n_frames": sum(s["n_frames"] for s in splits.values()),
            "n_boxes": n_boxes,
            "n_duplicate_clusters": len(duplicates),
            "n_flagged": len(report_rows),
        },
        "splits": splits,
        "classes": classes,
        "duplicates": sorted(duplicates, key=lambda d: d["dhash"]),
    }
    with FileSystems.create(output_path) as fh:
        fh.write(json.dumps(profile, indent=2, sort_keys=True).encode("utf-8"))


def _read_manifest(input_dir: str) -> list[dict[str, str]]:
    with FileSystems.open(FileSystems.join(input_dir, "manifest.csv")) as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))


def _query_counters(result) -> dict[str, int]:  # noqa: ANN001 - Beam PipelineResult
    metrics = result.metrics().query(MetricsFilter().with_namespace(COUNTER_NAMESPACE))
    return {m.key.metric.name: m.result for m in metrics["counters"]}


def run(options: ProfileOptions) -> int:
    """Execute the profiling pipeline; assumes all options are fully resolved (see ``main``)."""
    input_dir = options.input_dir
    profile_dir = FileSystems.join(input_dir, profiling.PROFILE_DIRNAME)
    if FileSystems.exists(profile_dir):
        FileSystems.delete([profile_dir])
    FileSystems.mkdirs(profile_dir)

    rows = _read_manifest(input_dir)
    manifest_sha = provenance.manifest_sha256(Path(input_dir) / "manifest.csv")

    with beam.Pipeline(options=options) as p:
        outputs = (
            p
            | "CreateFrames" >> beam.Create(rows)
            | "ProfileFrames"
            >> beam.ParDo(ProfileFrameDoFn(input_dir)).with_outputs(
                "corrupt", "boxes", main="records"
            )
        )
        split_stats = (
            outputs.records
            | "ToSplitMetrics" >> beam.Map(_split_metrics)
            | "CombinePerSplit" >> beam.CombinePerKey(StatsCombineFn())
        )
        class_stats = outputs.boxes | "CombinePerClass" >> beam.CombinePerKey(StatsCombineFn())
        duplicates = (
            outputs.records
            | "ToHashKeys" >> beam.Map(lambda r: (r["dhash"], r["image_relpath"]))
            | "GroupByHash" >> beam.GroupByKey()
            | "DuplicateClusters" >> beam.FlatMap(_duplicate_clusters)
        )
        flagged = outputs.records | "FlagFrames" >> beam.FlatMap(_flag_rows)
        report_rows = (flagged, outputs.corrupt) | "MergeReport" >> beam.Flatten()
        _ = (
            report_rows
            | "ToReportCsv" >> beam.Map(_report_csv_line)
            | "WriteReport"
            >> WriteToText(
                FileSystems.join(profile_dir, "quality_report"),
                file_name_suffix=".csv",
                shard_name_template="",
                header=",".join(REPORT_FIELDS),
            )
        )
        _ = (
            p
            | "TriggerAssembly" >> beam.Create([None])
            | "BuildProfile"
            >> beam.Map(
                _build_profile,
                split_stats=pvalue.AsDict(split_stats),
                class_stats=pvalue.AsList(class_stats),
                duplicates=pvalue.AsList(duplicates),
                report_rows=pvalue.AsList(report_rows),
                manifest_sha=manifest_sha,
                output_path=FileSystems.join(profile_dir, profiling.PROFILE_JSON),
            )
        )

    counters = _query_counters(p.result)
    corrupt = counters.get("corrupt_images", 0)
    logger.info(
        "profiled %s frames across %s splits | corrupt: %s",
        counters.get("frames_profiled", 0),
        len(SPLITS),
        corrupt,
    )
    if corrupt:
        # No stamp: the profile is not "current", so the next orchestrated run re-profiles.
        logger.error("%d corrupt/unreadable images — see profile/quality_report.csv", corrupt)
        return 1
    with FileSystems.create(FileSystems.join(profile_dir, profiling.STAMP_FILENAME)) as fh:
        fh.write(provenance.stamp_payload(manifest_sha, profiling.PROFILE_PARAMS).encode("utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    options = ProfileOptions(argv)
    # Settings-backed default, resolved at run time (never at class definition).
    options.input_dir = options.input_dir or str(settings.data.subset_dir)
    return run(options)


if __name__ == "__main__":
    raise SystemExit(main())
