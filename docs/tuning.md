# Tuning

A staged tuning grid picks the best training recipe for each model size (yolo26n, s, m) on the
VisDrone-VID subset: frame stride 20, with the raw categories merged into three classes (person,
vehicle, two-three-wheeler; see [data](data.md#annotation-format-and-3-class-merge)).

## Protocol

- **Selection on val.** Trials are compared on val mAP50-95. Test is read once, at the end, for the
  finalists and the zero-shot baselines.
- **Trials never register.** They log to the MLflow experiment `aerial-object-detection-tune` and
  never touch the registry or the champion alias. A finalist reaches production through the normal
  train → evaluate → gate path.
- **Staged greedy grid.** First the zero-shot baselines, then the current recipe per size. After
  that, each stage varies one or two factors as a small full grid and carries its winner forward.
- **Noise floor.** Three seeds of s@640. A trial wins only if it beats its stage's winner so far by
  more than the noise floor.

## Run a stage

```bash
make tune STAGE=smoke              # configs/tuning/smoke.yaml
make tune STAGE=stage2 PARALLEL=2  # two trials train at once
```

A stage file lists its trials. `defaults` apply to every trial, and a trial's own knobs win:

```yaml
defaults: {patience: 0, close_mosaic: 10}
trials:
  - {name: n640, model: yolo26n, imgsz: 640, epochs: 30, batch: 16}
  - {name: n640-zoom, model: yolo26n, imgsz: 640, epochs: 30, batch: 16, scale: [0.5, 1.5]}
```

`model`, `imgsz`, `epochs`, `batch` and `seed` become training flags. Every other knob is passed
to `model.train()` through `--set`. The runner starts one training process per trial, one at a
time unless `--parallel` is given. It then measures each trial's latency on its own, with nothing
else on the GPU. Run names are `<stage>-<trial>`. Parallel trials refuse knobs that change the
image list (`data`, `fraction`), because they would race on the shared label cache.

Each trial run records:

| Key | What |
|---|---|
| `val/mAP50-95`, `val/mAP50`, `val/precision`, `val/recall` | the selected weights (`best.pt`) on val |
| `val/mAP50-95/<class>` | per merged class |
| `val/mAP50-95_small` | COCO small-object AP ([evaluation](evaluation.md)) |
| `latency/p50_ms`, `latency/p95_ms`, `latency/mean_ms` | batch-1 predict at the trial's imgsz on the server GPU |
| `metrics/*` | ultralytics' per-epoch curves |

## Zero-shot baselines

`make zero-shot` scores the COCO-pretrained weights on val, with no training. COCO classes are
mapped onto the merged classes: person to person; car, bus and truck to vehicle; bicycle and
motorcycle to two-three-wheeler. Every other COCO class is dropped. Predictions use val's floor
and cap (conf 0.001, 300 boxes, applied before the mapping). When one box comes out under two
COCO classes that map to the same merged class, only the higher-confidence copy is kept. Scoring
goes through `score_detections`. Each baseline is a `zero-shot-<size><imgsz>` run in the tune
experiment. Values are in %; latency is the trial stub.

| Model | imgsz | val mAP50-95 | mAP50 | person | vehicle | two-three-wheeler | mAP50-95_small | p50 ms |
|---|---|---|---|---|---|---|---|---|
| yolo26n | 640 | 13.6 | 31.0 | 13.5 | 22.6 | 4.9 | 2.8 | 12.4 |
| yolo26n | 1280 | 20.0 | 42.2 | 19.8 | 30.8 | 9.4 | 6.7 | 18.6 |
| yolo26s | 640 | 18.5 | 38.3 | 16.7 | 30.3 | 8.6 | 4.8 | 12.8 |
| yolo26s | 1280 | 22.5 | 47.5 | 20.8 | 37.0 | 9.9 | 7.4 | 23.5 |
| yolo26m | 640 | 20.0 | 41.6 | 18.4 | 33.2 | 8.4 | 6.0 | 20.7 |
| yolo26m | 1280 | 22.5 | 47.9 | 21.8 | 37.1 | 8.5 | 8.1 | 33.4 |

## Noise floor (s@640, current recipe)

| Seed | val mAP50-95 | person | vehicle | two-three-wheeler |
|---|---|---|---|---|

## Stage 1: the current recipe per size

| Model | imgsz | val mAP50-95 | mAP50-95_small | p50 ms | best epoch |
|---|---|---|---|---|---|
