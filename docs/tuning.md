# Tuning

A staged tuning grid picks the best training recipe for each model size (yolo26n, s, m) on the
VisDrone-VID subset: frame stride 20, with the raw categories merged into three classes (person,
vehicle, two-three-wheeler; see [data](data.md#annotation-format-and-3-class-merge)).

## Protocol

- **Selection on val.** Trials are compared on val mAP50-95. Test is read once, at the end, for the
  shared recipe at 640 and 1280 (below). Test gaps are plain numbers, read against test's seed
  spread.
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
- **One recipe for every size.** A recipe setting is chosen once for n, s and m, on the mean val
  mAP50-95 over the three sizes with the win margin. Only the model and its pinned batch differ.
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

## Stage 4: optimizer and lr0 on s

`make tune STAGE=stage4` pins the optimizer for s@1280×30 against stage 2's s1280 row, which
ran `optimizer=auto` (AdamW, lr0 0.001429 on 8.4.72). Every arm also sets `momentum=0.9` and
`warmup_bias_lr=0.0`, as `auto` does, so only the optimizer and lr0 change. Values are in %.

| Optimizer | lr0 | val mAP50-95 | person | vehicle | two-three-wheeler | mAP50-95_small | last six | best epoch |
|---|---|---|---|---|---|---|---|---|
| **AdamW (auto)** | **0.001429** | **30.8** | 26.5 | 42.9 | 22.9 | 13.5 | 30.6 | 25 |
| AdamW | 0.0007 | 30.7 | 28.6 | 41.3 | 22.0 | 13.4 | 30.0 | 22 |
| AdamW | 0.003 | 30.5 | 26.3 | 42.6 | 22.6 | 14.1 | 30.2 | 25 |
| MuSGD | 0.01 | 31.1 | 27.9 | 42.2 | 23.3 | 12.3 | 30.4 | 9 |

`optimizer=auto` stays. MuSGD gains 0.3 pt, inside the win margin, from a `best.pt` at epoch 9.
Its last-six mean is 0.1 pt below auto's. The four arms span 0.6 pt, so lr0 barely matters
here.

## Stage 5: scale and mosaic on s

`make tune STAGE=stage5` crosses `scale` with `mosaic` for s@1280×30. The 0.5 × 1.0 cell is
stage 2's s1280 row. `scale: [0.75, 1.5]` is an absolute zoom range that only ever shrinks to
0.75 and can upscale, which is the full-frame stand-in for tile training (the SAHI paper's
upscaled crops). Values are in %.

| scale | mosaic | val mAP50-95 | person | vehicle | two-three-wheeler | mAP50-95_small | last six | best epoch |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 1.0 | 30.8 | 26.5 | 42.9 | 22.9 | 13.5 | 30.6 | 25 |
| 0.25 | 1.0 | 30.4 | 26.7 | 42.4 | 22.0 | 13.7 | 29.6 | 20 |
| **[0.75, 1.5]** | **1.0** | **31.4** | 27.5 | 43.4 | 23.2 | 14.9 | 31.0 | 27 |
| 0.5 | 0.5 | 31.2 | 27.7 | 43.3 | 22.5 | 13.7 | 30.8 | 27 |
| 0.25 | 0.5 | 30.5 | 26.6 | 42.1 | 22.8 | 13.1 | 29.9 | 19 |
| [0.75, 1.5] | 0.5 | 31.0 | 28.1 | 42.8 | 22.1 | 14.3 | 30.2 | 22 |

The zoom range wins: +0.6 pt over 0.5 × 1.0, just past the win margin. Its last-six mean
agrees (+0.4 pt), and it lifts mAP50-95_small by 1.4 pt at the same cost. Halving mosaic gains
0.4 pt alone (inside the margin) and nothing on top of the zoom. A narrower `scale` of 0.25
loses with either mosaic.

## Stages 6–7: close_mosaic and cls_pw on s

`make tune STAGE=stage6`, then `STAGE=stage7`, run on s's stage-5 winner (1280, 30 epochs,
`scale=[0.75, 1.5]`). Stage 6 keeps mosaic on to the last epoch. Stage 7 adds `cls_pw`,
ultralytics' inverse-frequency classification-loss weights. Values are in %.

| Setup | val mAP50-95 | person | vehicle | two-three-wheeler | mAP50-95_small | last six | best epoch |
|---|---|---|---|---|---|---|---|
| **stage 5 winner (`close_mosaic=10`, `cls_pw=0`)** | **31.4** | 27.5 | 43.4 | 23.2 | 14.9 | 31.0 | 27 |
| `close_mosaic=0` | 31.0 | 27.6 | 42.8 | 22.4 | 14.1 | 30.6 | 26 |
| `cls_pw=0.5` | 30.9 | 27.7 | 43.1 | 21.9 | 13.6 | 30.6 | 23 |
| `cls_pw=1.0` | 30.4 | 27.5 | 42.2 | 21.4 | 14.2 | 29.6 | 21 |

Both lose, so `close_mosaic=10` and `cls_pw=0` stay, and stage 6 needs no follow-up at 5 or 15.
`cls_pw` lowers the class it upweights most: two-three-wheeler falls 1.3 and 1.8 pt.

s's tuned recipe is the current recipe at imgsz 1280 with `scale=[0.75, 1.5]`.

## Confirmation stage: s's settings on n and m

`make tune STAGE=confirm` carries s's one change, `scale=[0.75, 1.5]`, to n and m at 1280×30.
Because m@1280 won stage 2, it adds one m trial with `freeze=11` (the whole backbone frozen).
Values are in %.

| Trial | val mAP50-95 | mAP50-95_small | last six | best epoch | GPU h |
|---|---|---|---|---|---|
| **n, current recipe (stage 2)** | **28.4** | 13.9 | 28.2 | 29 | 0.34 |
| n + zoom | 28.0 | 14.8 | 27.8 | 26 | 0.33 |
| **m, current recipe (stage 2)** | **32.9** | 15.9 | 31.6 | 19 | 0.96 |
| m + zoom | 32.6 | 16.1 | 31.3 | 17 | 0.97 |
| m + zoom + `freeze=11` | 31.3 | 15.9 | 31.0 | 13 | 0.63 |

The zoom doesn't carry over. It lowers n and m by 0.4 and 0.2 pt, on `best.pt` and on the
last-six mean, though it still lifts mAP50-95_small. A frozen backbone loses 1.5 pt.

## Final winners

The shared recipe is the ultralytics defaults at `imgsz=1280` and 30 epochs. s's zoom scores a
mean of 30.67 over the three sizes against 30.70 without it, inside the margin, so it drops out
and s's winner is `stage2-s1280`.

| Model | Trial run | val mAP50-95 | mAP50-95_small | p50 ms |
|---|---|---|---|---|
| yolo26n | `stage2-n1280` | 28.4 | 13.9 | 12.1 |
| yolo26s | `stage2-s1280` | 30.8 | 13.5 | 16.2 |
| yolo26m | `stage2-m1280` | 32.9 | 15.9 | 25.0 |

Against the ultralytics defaults the recipe changes only imgsz (640 → 1280, stage 2) and epochs
(100 → 30, stage 3). Batch 4/4/2 with `nbs=64` is the 8 GB GPU's ceiling. No augmentation,
optimizer, loss or freeze setting beat the defaults by the margin. The whole grid, stages 2 to
confirmation, took 20.9 GPU-hours.

## Evaluation-only arms

`make sahi-eval TRIALS="[--slice N] <trial run> ..."` scores a trial's `best.pt` two more ways, each as a
`<trial>-<arm>` run in the tune experiment. Arms never select. s's arms ran on
`stage5-s1280-zoom0.75-1.5`, its winner before the one-recipe rule.

- **`o2m`:** `model.val(end2end=False)`, the one-to-many head with NMS on the same weights, scored
  like the trial's own `val/` keys.
- **`sahi`:** SAHI 0.12.8 sliced inference (slices at the trial's imgsz, overlap 0.2, plus the
  full frame; class-aware NMS at IoU 0.5; floor 0.001; the top 300 per frame), against the same
  trial's full-frame predictions (`val_full/`). Both are scored through `score_detections`,
  overall and per source-resolution group (`val/3840px/...`). Latency is per frame, batch 1.
  `sahi` lives in the `tuning` uv group, outside every image.

Values are in %.

| Model | end2end (trial) | `end2end=False` | full frame | SAHI | SAHI − full: 3840 / 2720 / ≤1904 px | SAHI p50 ms |
|---|---|---|---|---|---|---|
| yolo26n | 28.4 | **30.2** | 28.5 | 29.8 | +8.5 / +1.6 / +0.3 | 95 |
| yolo26s | 31.4 | **32.6** | 31.2 | 31.6 | +4.6 / −0.3 / +0.9 | 150 |
| yolo26m | 32.9 | **34.0** | 32.6 | 31.9 | +4.0 / −2.7 / +0.6 | 260 |

- **The one-to-many head scores 1.2–1.8 pt higher** on the same weights for every size, more than
  the whole tuning grid gained after imgsz. Serving exports the end2end NMS-free head
  ([serving](serving.md)), so this is an input to the production default, not a recipe change.
- **SAHI pays off only on 3840-wide frames.** At ≤1904 px, the resolution test is mostly at,
  it gains 0.3–0.9 pt for 8–10× the latency (5.8 passes per val frame). For m the class-aware NMS
  merge costs more than slicing gains outside 3840. The serving re-entry condition, a gain beyond
  the noise floor at test-like resolutions within a named latency budget, is not met.
- **Smaller slices lose.** `--slice 640` (each slice upscaled 2× to imgsz, 18.3 passes) scores
  n/s/m 26.7/26.9/29.2 against 29.8/31.6/31.9 with 1280 slices, at twice the latency. It loses
  in every resolution group and on mAP50-95_small too, so training on 640 slices has no
  inference gain to build on.

## Slicing-aided fine-tuning

SAHI's training half, paired with its sliced inference. `make tune-sliced [STAGE=sliced-e8]`
cuts train into the windows the `sahi` arm infers on (1238 frames → 3300 slices; a box stays if a
tenth of it is inside) and retrains each final winner on frames + slices; val stays full frames.
Frames + slices are 3.7× an epoch, so `sliced` (30 epochs) runs 3.7× the steps and `sliced-e8`
(8 epochs, `close_mosaic=3`, `warmup_epochs=0.8`) the baselines' steps. Values are in %; each
cell is winner → `sliced` → `sliced-e8`.

| Model | SAHI | SAHI 3840 / 2720 / ≤1904 px | full frame | `end2end=False` |
|---|---|---|---|---|
| yolo26n | 29.8 → 29.4 → **30.7** | 15.9 / 33.7 / 30.9 → 14.5 / 35.6 / 31.4 (e8) | 28.5 → 28.2 → 28.4 | 30.2 → 29.7 → 29.9 |
| yolo26s | 31.6 → 30.5 → 31.4 | 15.1 / 36.0 / 32.5 → 19.6 / 36.3 / 31.7 (e8) | 31.2 → 30.2 → 30.0 | 32.6 → 31.4 → 31.7 |
| yolo26m | 31.9 → 31.3 → 31.7 | 19.8 / 35.6 / 33.8 → 21.9 / 35.3 / 33.1 (e8) | 32.6 → 31.5 → 31.5 | 34.0 → 33.5 → 32.9 |

- **At 30 epochs it overfits.** Every `sliced` run peaks at epoch 9–10 and loses on every read.
- **At equal steps only n's SAHI read clears the win margin** (+0.9). s and m tie on SAHI and
  lose 1.1–1.2 pt on full frames and the one-to-many head. The 3840 group gains for s and m
  (+4.5/+2.1, 49 frames) and the ≤1904 group doesn't.
- **No winner changes.** n's best, 30.7 at 97 ms, sits below s's full-frame
  `end2end=False` read (32.6) at a fraction of the cost (s's end2end p50 is 16 ms).
  Train has no 3840-wide frames, which may cap the 3840 gain. The two stages took ~8 GPU-hours.

## Test read

`make test-read TRIALS="<trial run> ..."` scores a trial's `best.pt` on test (337 frames, 17
sequences) with the served end2end head, as a `<trial>-test` run in the tune experiment;
`TRIALS="--o2m ..."` scores the one-to-many head as `<trial>-test-o2m`. It selects nothing. The
shared recipe is read at 640 (stage 1) and 1280 (stage 2) for every size. Values are in %; the
s@640 row is the mean of three seeds (22.5/22.9/22.8), and latency is the
trial's own. Per-class and small-object columns are the end2end head.

| Model | imgsz | val mAP50-95 | test mAP50-95 | test `end2end=False` | person | vehicle | two-three-wheeler | test mAP50-95_small | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| yolo26n | 640 | 21.0 | 18.5 | – | 9.4 | 36.7 | 9.4 | 11.8 | 11.0 |
| yolo26s | 640 | 26.3 | 22.7 | – | 14.1 | 41.8 | 12.4 | 14.3 | 10.3 |
| yolo26m | 640 | 28.6 | 25.1 | – | 16.8 | 45.0 | 13.5 | 15.2 | 12.0 |
| yolo26n | 1280 | 28.4 | 25.0 | 26.2 | 17.4 | 44.5 | 13.1 | 14.7 | 12.1 |
| yolo26s | 1280 | 30.8 | 26.6 | 27.8 | 19.0 | 46.3 | 14.6 | 16.3 | 16.2 |
| **yolo26m** | **1280** | 32.9 | 28.3 | **29.5** | 20.7 | 48.1 | 16.1 | 18.4 | 25.0 |

- **Val's ranking holds on test.** 1280 beats 640 for every size (n/s/m +6.5/+3.9/+3.2 against
  val's +7.4/+4.5/+4.3), m > s > n at both sizes, and n@1280 still matches m@640 (25.0 vs 25.1).
- **Test reads 2.5–4.6 pt below val**, the gap mostly widening with the val score.
- **The one-to-many head gains 1.2 pt on test for every size** (val: n/m +1.8/+1.1, s +1.2 on
  its zoom trial), so its val lead carries over.
- **Seed noise on test is 0.34 pt** (s@640's range), near val's 0.25, so the win margin carries.
- **The shared recipe beats the champion** (v2, s@640, 3 epochs, test 21.6) by 5.0 pt at s@1280.
- Not read on test yet: the third-party baseline.

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
- **Cost:** estimated at 22–40 GPU-hours, depending mostly on the epoch winner. With 30 epochs
  winning, stages 2 to confirmation took 20.9.

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

The serving default stays 0.25 during tuning. A tuned floor is not a one-off reading: if one
replaces 0.25, the pipeline computes it for every model version. That is decided with the
production default for model size and imgsz.
