# VisDrone-VID dataset

A small, reproducible YOLO detection dataset built from the **VisDrone-VID** (video object
detection) benchmark: drone-captured video frames with the 10 raw object categories merged into 3.

```mermaid
flowchart LR
  raw["VisDrone2019-VID-{train,val,test-dev}/<br/>sequences/ + annotations/"]
  raw -->|"ingest pipeline<br/>(frame-stride, 3-class merge)"| subset["&lt;DATA__SUBSET_DIR&gt;/<br/>images/ · labels/ · demo/ · manifest.csv"]
  subset -->|validate| gate["pass / fail-fast"]
  subset -->|"dataset YAML stamped by the consumer"| yolo["ultralytics train / val"]
```

## Download

The VID ZIPs are hosted on **Google Drive / Baidu Yun** (linked from the
[VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) repo). Download `VisDrone2019-VID-train`,
`VisDrone2019-VID-val`, and `VisDrone2019-VID-test-dev`, unzip, and point the converter at the parent
directory. Each split is already a set of **extracted frames** (not video files):

```
<DATA__RAW_DIR>/
  VisDrone2019-VID-train/
    sequences/<seq>/0000001.jpg, 0000002.jpg, ...
    annotations/<seq>.txt
  VisDrone2019-VID-val/
    sequences/ ... · annotations/ ...
  VisDrone2019-VID-test-dev/
    sequences/ ... · annotations/ ...
```

Set the locations (via the `export` or `.env`):

```bash
export DATA__RAW_DIR=/abs/path/to/VisDrone-VID    # holds VisDrone2019-VID-{train,val,test-dev}/
export DATA__SUBSET_DIR=/abs/path/to/subset       # where the ingest pipeline writes
```

## Annotation format and 3-class merge

Each annotation line is comma-separated:

```
0:frame_index
1:target_id
2:bbox_left
3:bbox_top
4:bbox_width
5:bbox_height
6:score
7:object_category
8:truncation
9:occlusion
```

`bbox_*` are absolute pixels with `(left, top)` the top-left corner. In the **ground truth**,
`score` is `1` (counts) or `0` (ignored — skipped). The 10 raw categories collapse to 3:

| Merged class (id)        | VisDrone raw categories                                   |
|--------------------------|-----------------------------------------------------------|
| `person` (0)             | pedestrian (1), people (2)                                |
| `vehicle` (1)            | car (4), van (5), truck (6), bus (9)                      |
| `two-three-wheeler` (2)  | bicycle (3), tricycle (7), awning-tricycle (8), motor (10)|
| *dropped*                | ignored regions (0), others (11), and any `score == 0` row |

The third class is named `two-three-wheeler` because it includes the 3-wheeled tricycle and
awning-tricycle alongside the 2-wheeled bicycle and motor(cycle). Each surviving box is converted to
a YOLO line `<cls> <xc> <yc> <w> <h>` with the box **center** and size normalized to `[0, 1]` and
clipped to the frame.

## Ignored regions: dropped, not masked

VisDrone marks crowded / low-resolution areas as **ignored regions** (`category 0`, `score 0`) so
detections there are neither rewarded nor penalized during the official evaluation. This converter
**drops** those boxes (it does not paint the pixels out).

## Building the subset

```bash
make ingest
# OR with custom flags:
uv run python -m mlops_cv.pipelines.ingest_pipeline --runner DirectRunner \
  --frame-stride 10
```

Subset building is the Beam **ingestion pipeline** ([batch-pipeline.md](batch-pipeline.md)). It
shrinks the set with `--frame-stride N` — keeping every Nth frame to drop the near-duplicate
consecutive frames of a video. Flags:

- `--frame-stride N` — keep every Nth frame in train, val, and test splits (default 20).
- `--sequences NAME ...` — restrict to specific sequences (default: all) for a quick smoke.
- `--demo-clips PATH` — demo-clips YAML for the demo store (default
  `configs/demo_clips.yaml`; `none` disables).
- `--raw-dir / --output-dir` — override the `Settings.data.*` defaults.

It writes a self-contained YOLO tree to `DATA__SUBSET_DIR` (default: `data/visdrone-vid-small`
inside the repo; any absolute path works — training, validation, evaluation, and the CT DAG all
read the same setting, and the directory's basename becomes the MLflow dataset name):

```
<DATA__SUBSET_DIR>/
  images/{train,val,test}/<seq>_<NNNNNNN>.jpg
  labels/{train,val,test}/<seq>_<NNNNNNN>.txt
  demo/{images,labels}/<sequence>/<NNNNNNN>.{jpg,txt}   # full-rate demo-clip store (evaluation)
  profile/{profile.json,quality_report.csv}             # written by the profiling pipeline
  manifest.csv
  VisDrone-VID-merged.yaml                              # stamped by each consumer, see below
```

Ingest writes no dataset YAML. Each ultralytics consumer (training, evaluation, optimize; the edge
gate writes a test-only `-test.yaml`) stamps the committed template next to the data with `path:`
set to its own absolute `DATA__SUBSET_DIR`, so the YAML always names the reading machine's copy.
For an ad-hoc `yolo` run, stamp it first:

```bash
uv run python -c "from mlops_cv.config import get_settings; \
from mlops_cv.data.subset import dataset_yaml; print(dataset_yaml(get_settings()))"
yolo val model=yolo26s.pt data=<DATA__SUBSET_DIR>/VisDrone-VID-merged.yaml
```

The committed [`configs/datasets/VisDrone-VID-merged.yaml`](../configs/datasets/VisDrone-VID-merged.yaml)
is the human-readable source of truth for the class names.

## Validating the dataset

```bash
uv run python -m mlops_cv.data.validate     # exits non-zero on any failure
```

`validate_dataset()` aggregates every failure and either raises `DatasetValidationError` or returns
a `Report` whose `summary()` is a one-line string. It checks: train, val, and test splits non-empty; exact
image↔label pairing per split; every class id in `{0, 1, 2}`; every bbox coordinate in `[0, 1]` with
positive width/height; and each merged class present. The function is intended to be reused as a
fail-fast data-validation gate in the orchestration pipeline.

## Manifest

`manifest.csv` has one row per kept frame, for downstream dataset tracking and the
class-distribution check:

```
split,sequence,frame_index,image_relpath,label_relpath,width,height,n_boxes,n_person,n_vehicle,n_two_three_wheeler
```

## Moving the data to an object store

The ultralytics consumers read a local directory, and each stamps its own dataset YAML, so a
subset copied onto another host works there unchanged. Move it when a second compute host needs
it (a cloud GPU, a second trainer, the Mac), for durability (the store must sit on separate
hardware: the local RustFS shares this disk, so it is no backup), or when the data outgrows the
disk. Ingest and profiling can write the store directly
([batch-pipeline.md](batch-pipeline.md#cloud-migration)); the ultralytics consumers still need the
pull below.

Any object store with a sync CLI works. Give the data a bucket (or container) of its own, never
the MLflow artifact one, and keep these rules whichever tool you use:

- **Push** `DATA__SUBSET_DIR` from the host that ran ingest, again after each re-ingest, deleting
  remote files the subset no longer has. Push the raw dataset the same way if another host must
  re-ingest.
- **Pull** into the other host's `DATA__SUBSET_DIR`, deleting local files the store doesn't have.
  Keep the basename: it is the MLflow dataset name.
- **Exclude** `*.npy`, `*.cache` and `*.yaml` both ways: ultralytics' caches and the per-host
  stamped YAMLs. The delete must leave excluded files alone, or a pull wipes that host's caches
  and stamps.

`manifest.csv` comes along, so training against the same MLflow server finds its data version,
and `profile/` with its stamp stays current. Pull direction shown; swap the arguments to push:

| Store | Command |
|---|---|
| S3 or S3-compatible | `aws s3 sync <uri> <dir> --delete --exclude '*.npy' --exclude '*.cache' --exclude '*.yaml'` |
| GCS | `gcloud storage rsync -r --delete-unmatched-destination-objects --exclude '.*\.(npy\|cache\|yaml)$' <uri> <dir>` |
| Azure Blob | `azcopy sync <url> <dir> --delete-destination=true --exclude-pattern '*.npy;*.cache;*.yaml'` |
| Any (one tool) | `rclone sync <remote:path> <dir> --exclude '*.{npy,cache,yaml}'` |
