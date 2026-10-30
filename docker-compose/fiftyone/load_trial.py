"""Load a labeling trial's auto-labels into the FiftyOne App: ``make autolabel-view RUN=<id>``.

Runs inside the App container, so the client always matches the server. The trial file sits at
``<autolabel dir>/<set>/trials/<run name>.json``, beside the set's ``ground_truth.json``.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

GROUND_TRUTH = "ground_truth"


def dataset_name(scoring_set: str) -> str:
    return f"visdrone-vid-{scoring_set}"


def field_prefix(run_name: str, scoring_set: str) -> str:
    """The run name without its set suffix, as a FiftyOne field name (no dots or dashes)."""
    return re.sub(r"\W", "_", run_name.removesuffix(f"-{scoring_set}"))


def frame_fields(file_name: str) -> dict[str, Any]:
    """``<sequence>_<frame>.jpg`` -> its sequence and frame number."""
    sequence, frame = Path(file_name).stem.rsplit("_", 1)
    return {"sequence": sequence, "frame_number": int(frame)}


def with_prompt(annotation: dict[str, Any]) -> dict[str, Any]:
    """The detector prompt moved from ``attributes.label``, which clashes with the class label."""
    attributes = dict(annotation.get("attributes", {}))
    if "label" in attributes:
        attributes["prompt"] = attributes.pop("label")
    return annotation | {"attributes": attributes}


def detector_boxes(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The trial's annotations with the detector's box as ``bbox`` and no mask.

    A sequence-mode auto-label on a tracked frame has no detector box, so it's left out.
    """
    return [
        {k: v for k, v in a.items() if k not in ("segmentation", "det_bbox")}
        | {"bbox": a["det_bbox"]}
        for a in annotations
        if "det_bbox" in a
    ]


def copy_cropped_masks(fouc: Any) -> None:
    """Copy each mask crop: FiftyOne 1.22.1 slices the decoded frame, keeping all of it alive.

    ponytail: patches a private FiftyOne function; recheck on a FiftyOne bump.
    """
    crop = fouc._coco_segmentation_to_mask
    fouc._coco_segmentation_to_mask = lambda *a: None if (m := crop(*a)) is None else m.copy()


def create(fo: Any, name: str, set_dir: Path) -> Any:
    """The set's frames with ``ground_truth`` and ``ignored``, loaded once."""
    import fiftyone.utils.coco as fouc

    gt = json.loads((set_dir / "ground_truth.json").read_text())
    ignored = json.loads((set_dir / "ignored.json").read_text())
    ds = fo.Dataset(name, persistent=True)
    ds.add_samples(
        fo.Sample(
            filepath=str(set_dir / "images" / im["file_name"]),
            metadata=fo.ImageMetadata(width=im["width"], height=im["height"]),
            **frame_fields(im["file_name"]),
        )
        for im in gt["images"]
    )
    fouc.add_coco_labels(ds, GROUND_TRUTH, gt["annotations"], gt["categories"])
    fouc.add_coco_labels(ds, "ignored", ignored["annotations"], ignored["categories"])
    return ds


def main(trial: Path) -> None:
    import fiftyone as fo
    import fiftyone.utils.coco as fouc

    copy_cropped_masks(fouc)
    start = time.perf_counter()
    set_dir = trial.parent.parent
    scoring_set = set_dir.name
    name = dataset_name(scoring_set)
    ds = fo.load_dataset(name) if fo.dataset_exists(name) else create(fo, name, set_dir)

    coco = json.loads(trial.read_text())
    frames = [Path(p).name for p in ds.values("filepath")]
    if [im["file_name"] for im in coco["images"]] != frames:
        sys.exit(f"{trial.name}: its images don't match dataset {name}'s frames")

    prefix = field_prefix(trial.stem, scoring_set)
    annotations, categories = [with_prompt(a) for a in coco["annotations"]], coco["categories"]
    fields = {f"{prefix}_det": ("detections", detector_boxes(annotations))}
    if any("segmentation" in a for a in annotations):  # the champion row has no masks
        fields[f"{prefix}_seg"] = ("segmentations", annotations)
    for field, (label_type, anns) in fields.items():
        eval_key = f"eval_{field}"
        if ds.has_evaluation(eval_key):
            ds.delete_evaluation(eval_key)
        if ds.has_sample_field(field):
            ds.delete_sample_field(field)
        fouc.add_coco_labels(ds, field, anns, categories, label_type=label_type)
        # Browsing only: these flags count every confidence and match unlike the validator.
        ds.evaluate_detections(field, GROUND_TRUTH, eval_key=eval_key, iou=0.5)
    print(f"{name}: {', '.join(fields)} in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
