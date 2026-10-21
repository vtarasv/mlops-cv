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
- **Win margin: 0.5 pt** of overall val mAP50-95. That is about twice the noise floor (three
  seeds of s@640, below). The floor is a lower bound: data order is fixed and every seed is
  scored on the same val, so the margin sits above it. One margin covers every size and imgsz.
  - A challenger inside the margin loses, and the stage's winner so far stays.
  - When two challengers both win but sit within the margin of each other, the cheaper setup
    (lower trial GPU time) carries forward.
  - Per-class scores and mAP50-95_small are recorded only. They never pick a winner.
- **Val stays at frame stride 20** (146 frames, 7 sequences). Adjacent frames are near-duplicates
  and the sequence is the independent unit, so a denser val would add frames without adding
  independent evidence.

## Run a stage

```bash
make tune STAGE=smoke  # configs/tuning/smoke.yaml
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
time. It then measures each trial's latency on its own, with nothing else on the GPU. Run names
are `<stage>-<trial>`.

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

`make tune STAGE=stage1` runs the noise floor and stage 1. Every trial uses the current recipe
(30 epochs, AdamW from `optimizer=auto`, `nbs=64`) with `patience=0`, `close_mosaic=10`, a pinned
batch and `workers=0`. On 8.4.72 dataloader workers ignore the seed. In the main process the seed
varies weight init and augmentation, but the data order stays fixed, so the spread is a lower
bound. Values are in %.

| Seed | val mAP50-95 | person | vehicle | two-three-wheeler | mAP50-95_small |
|---|---|---|---|---|---|
| 0 | 26.25 | 22.5 | 37.9 | 18.3 | 11.2 |
| 1 | 26.17 | 23.4 | 37.1 | 18.0 | 11.2 |
| 2 | 26.42 | 23.7 | 38.0 | 17.5 | 10.6 |
| range | 0.25 | 1.2 | 0.9 | 0.8 | 0.6 |

The standard deviation of the overall score is 0.13 pt. A single class moves 4–5× more than the
overall score.

## Stage 1: the current recipe per size

Batch 16 for n and s and batch 8 for m, so `accumulate` is 4 and 8: about 19 optimizer steps per
epoch for every size. The s row is the mean of the three seeds. Per-epoch time is with
`workers=0`, which leaves loading on one CPU core and the GPU about half idle.

| Model | imgsz | val mAP50-95 | mAP50-95_small | p50 ms | best epoch | s/epoch |
|---|---|---|---|---|---|---|
| yolo26n | 640 | 21.0 | 8.8 | 11.0 | 28 | 24.8 |
| yolo26s | 640 | 26.3 | 11.0 | 10.3 | 24, 25, 29 | 30.5 |
| yolo26m | 640 | 28.6 | 12.5 | 12.0 | 28 | 41.3 |

The best epoch falls between 24 and 29 of 30. Over the last six epochs the curve moves within the
noise. Whether more epochs help is the grid's call.

Two trials at once don't fit in 8 GB: two n@640 trials at batch 16 pushed the loss's label
assignment (`TaskAlignedAssigner`) onto the CPU on every batch, about 10× slower than running
them one after another. Even alone, s@640 at batch 16 reaches the 8 GB limit and fell back to the
CPU on 37 batches across its three seeds.

## Stage 2: imgsz per size

`make tune STAGE=stage2` runs imgsz 960 and 1280 for every size against its stage 1 row at 640
(30 epochs, pinned batch, `workers=4`). Values are in %. The 640 rows are repeated from stage 1.

| Model | imgsz | val mAP50-95 | person | vehicle | two-three-wheeler | mAP50-95_small | p50 ms | best epoch | s/epoch |
|---|---|---|---|---|---|---|---|---|---|
| yolo26n | 640 | 21.0 | 17.3 | 32.9 | 12.8 | 8.8 | 11.0 | 28 | 24.8 |
| yolo26n | 960 | 26.0 | 21.7 | 38.6 | 17.6 | 12.4 | 10.7 | 27 | 23.8 |
| **yolo26n** | **1280** | **28.4** | 24.4 | 41.5 | 19.5 | 13.9 | 12.1 | 29 | 40.6 |
| yolo26s | 640 | 26.3 | 23.2 | 37.7 | 17.9 | 11.0 | 10.3 | 24, 25, 29 | 30.5 |
| yolo26s | 960 | 28.9 | 25.0 | 41.9 | 19.8 | 14.3 | 12.4 | 25 | 39.2 |
| **yolo26s** | **1280** | **30.8** | 26.5 | 42.9 | 22.9 | 13.5 | 16.2 | 25 | 64.9 |
| yolo26m | 640 | 28.6 | 24.5 | 40.6 | 20.7 | 12.5 | 12.0 | 28 | 41.3 |
| yolo26m | 960 | 31.4 | 27.7 | 44.7 | 21.9 | 15.2 | 16.6 | 19 | 64.9 |
| **yolo26m** | **1280** | **32.9** | 28.2 | 44.3 | 26.1 | 15.9 | 25.0 | 19 | 115.0 |

1280 wins for every size, by 1.4–2.5 pt over 960, well past the win margin. n@1280 (28.4)
matches m@640 (28.6) at about the same latency. Both m trials peak at epoch 19, before mosaic
closes. m@1280's best epoch (32.8) is a one-epoch spike over a plateau near 31.6, and on the
mean of its last six epochs it leads m@960 by only 0.5 pt (31.6 vs 31.0), right at the margin. Its win still stands under the
protocol, which scores `best.pt`. Because m@1280 won, the confirmation stage runs `freeze=11`.

## Stage 3: epochs per size

`make tune STAGE=stage3` runs 60 and 100 epochs for every size at 1280 against stage 2's 30-epoch
row. The last-six column is the mean val mAP50-95 over the run's final six epochs, a check on
`best.pt` spikes. Values are in %.

| Model | epochs | val mAP50-95 | person | vehicle | two-three-wheeler | mAP50-95_small | last six | best epoch | GPU h |
|---|---|---|---|---|---|---|---|---|---|
| **yolo26n** | **30** | **28.4** | 24.4 | 41.5 | 19.5 | 13.9 | 28.2 | 29 | 0.34 |
| yolo26n | 60 | 28.7 | 24.2 | 42.3 | 19.6 | 14.3 | 28.2 | 48 | 0.69 |
| yolo26n | 100 | 28.1 | 23.7 | 40.5 | 20.2 | 12.1 | 27.5 | 21 | 1.15 |
| **yolo26s** | **30** | **30.8** | 26.5 | 42.9 | 22.9 | 13.5 | 30.6 | 25 | 0.54 |
| yolo26s | 60 | 30.1 | 25.5 | 42.7 | 22.2 | 13.3 | 28.9 | 38 | 1.09 |
| yolo26s | 100 | 30.4 | 25.9 | 41.9 | 23.4 | 12.0 | 28.0 | 38 | 1.85 |
| **yolo26m** | **30** | **32.9** | 28.2 | 44.3 | 26.1 | 15.9 | 31.6 | 19 | 0.96 |
| yolo26m | 60 | 32.0 | 28.9 | 44.6 | 22.5 | 16.1 | 30.0 | 29 | 1.93 |
| yolo26m | 100 | 31.8 | 27.4 | 44.7 | 23.2 | 16.1 | 28.4 | 25 | 3.32 |

30 epochs stays for every size. n at 60 epochs gains 0.3 pt, inside the win margin, and every
other longer run loses. Longer runs peak before the end and then fall: the last-six mean drops by
0.7–3.2 pt from 30 to 100 epochs, the signature of overfitting 1238 training frames rather than
of undertraining. Per-epoch time doesn't change with the budget (n/s/m about 41/66/118 s).

## The grid

Each stage is a full grid of its own factors, run against the winner so far. Stages 2 and 3 run
every size. Stages 4–7 run on s only, and the confirmation stage carries s's settings to n and m.

| Stage | Sizes | Factor | New levels (winner so far) |
|---|---|---|---|
| 2 | n, s, m | imgsz | 960, 1280 (640) |
| 3 | n, s, m | epochs | 60, 100 (30) |
| 4 | s | optimizer, lr0 | AdamW 0.0007, AdamW 0.003, MuSGD 0.01 (AdamW 0.001429) |
| 5 | s | `scale` × `mosaic` | {0.5, 0.25, (0.75, 1.5)} × {1.0, 0.5} (0.5 × 1.0) |
| 6 | s | `close_mosaic` | 0 (10) |
| 7 | s | `cls_pw` | 0.5, 1.0 (0) |
| Confirmation | n, m | s's settings from stages 4–7 | `freeze=11` on m, only if m@1280 won stage 2 |

- **Every trial:** seed 0, `patience=0`, `close_mosaic=10` (except in stage 6), `nbs=64` and the
  pinned batch below. Every trial trains on the subset's own manifest, so there are no background
  images and no derived lists.
- **Workers:** from stage 2 on, trials use the `Settings` default of 4 workers. That is 34% faster
  than `workers=0` at s@1280 (66.7 vs 89.5 s/epoch), and 8 workers is no faster (GPU at 96%). With
  workers the seed no longer varies augmentation, but each setup still gets one augmentation draw
  from the same distribution, so the noise floor and the win margin still hold.
- **Confirmation:** a size keeps s's settings only if they beat that size's stage-3 winner by
  the win margin.
- **Evaluation-only arms** run on the three final winners and never select: SAHI sliced inference
  (slices at the trial's imgsz, overlap 0.2, plus the full frame), reported by source resolution,
  and `val(end2end=False)` (the one-to-many head with NMS, on the same weights).
- **Cost:** about 22–40 GPU-hours, depending mostly on the epoch winner. That is an estimate, not a
  cap.

### Pinned batch per setup

These are measured ceilings. One step up, s and m run out of memory, and n's label assignment
falls back to the CPU at 2–7× the epoch time. At 1280 ultralytics' first-epoch halve-and-retry
crashes instead of recovering (`CUBLAS_STATUS_ALLOC_FAILED`), so the pin is required.

| Model | 640 | 960 | 1280 |
|---|---|---|---|
| yolo26n | 16 | 8 | 4 |
| yolo26s | 16 | 8 | 4 |
| yolo26m | 8 | 4 | 2 |

`nbs=64` keeps about 19 optimizer steps per epoch at every setup. BatchNorm still sees the small
batch, and 8 GB can't avoid that confound. With `workers=0`, an epoch took n/s/m 40.8/59.5/80.9 s
at 960 and 65.4/89.5/138.3 s at 1280.

## Confidence floor

The floor never ranks trials. Ranked metrics (mAP, AR, the grid's val mAP50-95) are scored at
conf 0.001 and 300 boxes per frame, and a higher floor can only lower them. Point metrics
(precision, recall, F1, false positives per frame, the anomaly rule's counts, prediction health,
tracker metrics) exist only at a floor. The logged `val/precision` and `val/recall` sit at
ultralytics' max-F1 confidence at IoU 0.5, not at the serving floor. Scored at the serving default
0.25, the champion reads about 3.3 pt lower on mAP50-95 from truncation alone, so online accuracy
is scored at a floor of 0.01 or lower.

The serving default stays 0.25 during tuning. Each finalist's max-F1 confidence is read at the
final test read, on val and on test, and recorded here. Whether one replaces 0.25 is decided with
the production default for model size and imgsz.
