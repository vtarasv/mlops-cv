# Tuning: YOLO26 on VisDrone-VID

These experiments look for the best way to train YOLO26 n, s and m on this repo's VisDrone-VID
subset, and check a few ideas on top of it: sliced inference, sliced training, a denser train split,
and a comparison with a public VisDrone checkpoint. The [summary](#summary) at the end lists what
was decided. Scores are mAP50-95 in %, on val unless a section says test.

## How the experiments work

- **Data.** The subset keeps every 20th video frame: train 1238 frames from 56 videos, val 146
  from 7, test 337 from 17. The ten VisDrone categories are merged into three classes: person,
  vehicle and two-three-wheeler ([data](data.md#annotation-format-and-3-class-merge)).
- **Trial.** One training run with one set of settings. Trials log to the MLflow experiment
  `aerial-object-detection-tune` and are never registered as model versions.
- **Score.** A trial is judged by the val mAP50-95 of its best checkpoint (`best.pt`). Per-class AP
  and mAP50-95_small (AP on objects under 32×32 px) are recorded but never decide.
- **Win margin: 0.5 pt.** Training one setup with three seeds moved its score by 0.25 pt (the
  [noise floor](#2-noise-floor-one-setup-three-seeds)). A new setting counts as better only if it
  beats the current best by 0.5 pt; otherwise the current setting stays. When two new settings both
  win and sit within 0.5 pt of each other, the cheaper one wins.
- **A few settings at a time.** Each stage changes one or two settings, keeps the best and passes
  it on. Stages 2–3 run on every size. Stages 4–7 run on s only to save GPU time, and a
  confirmation stage then tries s's changes on n and m. A setting is kept only if it helps all
  three sizes on average, so every size ends up with one shared recipe.
- **Test is read once, at the end.** It never picks a winner; it checks that val's picks hold.
- **Every trial** uses seed 0, `patience=0` (no early stopping), `close_mosaic=10` (mosaic off for
  the last 10 epochs, the ultralytics default) and the largest batch that fits the 8 GB GPU (table
  below), on ultralytics 8.4.72.

### Hardware

Every trial, latency reading and score was measured on one laptop:

| Part | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop, 8 GB, driver 615.71.09 |
| CPU | Intel Core Ultra 9 275HX, 24 cores |
| RAM | 32 GB |
| Software | PyTorch 2.12.1 (CUDA 13.0), ultralytics 8.4.72 |

GPU-hours and latencies are for this GPU and won't carry over to another one.

Clarification for table columns:

| Column | Meaning |
|---|---|
| last six | mean val score over the final six epochs: shows whether `best.pt` is a one-epoch spike |
| best epoch | the epoch `best.pt` comes from |
| GPU h | training time |
| p50 ms | median batch-1 predict latency on the server GPU at the trial's imgsz |

### Batch size per setup

These are the largest batches that fit in 8 GB. One step up, s and m run out of memory and n gets
2–7× slower (the loss's box matching falls back to the CPU). At 1280, ultralytics' automatic
batch halving crashes instead of recovering, so the batch must be set. `nbs=64` (the default)
accumulates gradients up to 64 images, so every setup takes about the same number of optimizer
steps per epoch.

| Model | 640 | 960 | 1280 |
|---|---|---|---|
| yolo26n | 16 | 8 | 4 |
| yolo26s | 16 | 8 | 4 |
| yolo26m | 8 | 4 | 2 |

## How to run

Start MLflow once, then run a stage file:

```bash
make mlflow-up
uv run python -m mlops_cv.tuning.tune configs/tuning/yolo26-visdrone-vid/stage2.yaml
# the same, through make:  make tune STAGE=yolo26-visdrone-vid/stage2
```

A stage file lists its trials. `defaults` apply to every trial, and a trial's own keys win:

```yaml
defaults: {patience: 0, close_mosaic: 10, epochs: 30, seed: 0}
trials:
  - {name: n1280, model: yolo26n, imgsz: 1280, batch: 4}
  - {name: n1280-zoom, model: yolo26n, imgsz: 1280, batch: 4, scale: [0.75, 1.5]}
```

`model`, `imgsz`, `epochs`, `batch` and `seed` become training CLI flags; every other key goes to
`model.train()` as is. Trials run one after another, because two at once don't fit in 8 GB. When
they finish, the runner measures each trial's latency with nothing else on the GPU. A run is named
`<stage file>-<trial>`, such as `stage2-n1280`. Each trial run records:

| Key | What |
|---|---|
| `val/mAP50-95`, `val/mAP50`, `val/precision`, `val/recall` | `best.pt` on val |
| `val/mAP50-95/<class>` | per merged class |
| `val/mAP50-95_small` | COCO small-object AP ([evaluation](evaluation.md)) |
| `latency/p50_ms`, `latency/p95_ms`, `latency/mean_ms` | batch-1 predict at the trial's imgsz |
| `metrics/*` | ultralytics' per-epoch curves |

The other experiment commands live in `mlops_cv.tuning`. Each section below gives the command that
produced its table. Stage files are in `configs/tuning/yolo26-visdrone-vid/`.

## 1. Zero-shot baselines

**What:** the COCO-pretrained yolo26n, s and m, with no training on VisDrone, at imgsz 640 and
1280. COCO classes are mapped onto ours: person to person; car, bus and truck to vehicle; bicycle
and motorcycle to two-three-wheeler. Every other COCO class is dropped. When one box comes out under
two COCO classes that map to the same class, only the higher-confidence copy is kept.

**Why:** the starting point. It shows how much training on VisDrone adds.

```bash
uv run python -m mlops_cv.tuning.zero_shot   # runs zero-shot-<size><imgsz>
```

| Model | imgsz | val mAP50-95 | mAP50 | person | vehicle | two-three-wheeler | mAP50-95_small | p50 ms |
|---|---|---|---|---|---|---|---|---|
| yolo26n | 640 | 13.6 | 31.0 | 13.5 | 22.6 | 4.9 | 2.8 | 12.4 |
| yolo26n | 1280 | 20.0 | 42.2 | 19.8 | 30.8 | 9.4 | 6.7 | 18.6 |
| yolo26s | 640 | 18.5 | 38.3 | 16.7 | 30.3 | 8.6 | 4.8 | 12.8 |
| yolo26s | 1280 | 22.5 | 47.5 | 20.8 | 37.0 | 9.9 | 7.4 | 23.5 |
| yolo26m | 640 | 20.0 | 41.6 | 18.4 | 33.2 | 8.4 | 6.0 | 20.7 |
| yolo26m | 1280 | 22.5 | 47.9 | 21.8 | 37.1 | 8.5 | 8.1 | 33.4 |

**Result:** 1280 adds 4–6 pt for every size, and untrained m is no better than s at 1280. Training
(stage 1) adds 7–9 pt at 640.

## 2. Noise floor: one setup, three seeds

**What:** s at imgsz 640 and 30 epochs on the default recipe, trained three times with seeds 0, 1
and 2. It uses `workers=0`, because on 8.4.72 the seed reaches augmentation only without worker
processes. The data order stays fixed, so this spread is a lower bound.

**Why:** to know how far a score moves by chance, so that a real gain can be told from noise.

```bash
uv run python -m mlops_cv.tuning.tune configs/tuning/yolo26-visdrone-vid/stage1.yaml   # also stage 1
```

| Seed | val mAP50-95 | person | vehicle | two-three-wheeler | mAP50-95_small |
|---|---|---|---|---|---|
| 0 | 26.25 | 22.5 | 37.9 | 18.3 | 11.2 |
| 1 | 26.17 | 23.4 | 37.1 | 18.0 | 11.2 |
| 2 | 26.42 | 23.7 | 38.0 | 17.5 | 10.6 |
| range | 0.25 | 1.2 | 0.9 | 0.8 | 0.6 |

**Result:** the overall score moves 0.25 pt (standard deviation 0.13), so a win must clear 0.5 pt.
A single class moves 4–5× more, which is why per-class scores never decide.

## 3. Stage 1: the starting recipe per size

**What:** n, s and m on the repo's starting recipe: imgsz 640, 30 epochs and the ultralytics
defaults (`optimizer=auto` picks AdamW with lr0 0.001429). The s row is the mean of the three
seeds above.

**Why:** the reference each size's later stages must beat.

Same command as the noise floor (`stage1.yaml`).

| Model | imgsz | val mAP50-95 | mAP50-95_small | p50 ms | best epoch | s/epoch |
|---|---|---|---|---|---|---|
| yolo26n | 640 | 21.0 | 8.8 | 11.0 | 28 | 24.8 |
| yolo26s | 640 | 26.3 | 11.0 | 10.3 | 24, 25, 29 | 30.5 |
| yolo26m | 640 | 28.6 | 12.5 | 12.0 | 28 | 41.3 |

**Result:**
- Training adds 7–9 pt over zero-shot.
- The best epoch lands between 24 and 29 of 30.
- Two trials at once are about 10× slower than one after another: they push the loss's box
  matching onto the CPU. So trials run one at a time.
- From stage 2 on, trials use 4 dataloader workers, which is 34% faster than `workers=0`.

## 4. Stage 2: image size

**What:** imgsz 960 and 1280 for every size, against stage 1's 640. 30 epochs each.

**Why:** VisDrone objects are tiny. At 640, 22% of train boxes are under 8 px on their short side;
at 1280, 1.7% are. A larger input gives each object more pixels.

```bash
uv run python -m mlops_cv.tuning.tune configs/tuning/yolo26-visdrone-vid/stage2.yaml
```

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

**Result:**
- 1280 wins for every size, by 1.4–2.5 pt over 960.
- n at 1280 (28.4) matches m at 640 (28.6) at about the same latency.
- m at 1280 peaks at epoch 19 with a one-epoch spike. On the last-six mean it leads m at 960 by
  only 0.5 pt (31.6 vs 31.0).

## 5. Stage 3: epochs

**What:** 60 and 100 epochs for every size at 1280, against stage 2's 30.

**Why:** the ultralytics default is 100 epochs; 30 was the repo's guess.

```bash
uv run python -m mlops_cv.tuning.tune configs/tuning/yolo26-visdrone-vid/stage3.yaml
```

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

**Result:** 30 epochs stays. Only n at 60 gains (+0.3, inside the margin). Longer runs peak early
and then decline: the last-six mean falls 0.7–3.2 pt from 30 to 100 epochs, so 1238 frames are
being overfitted, not undertrained.

## 6. Stage 4: optimizer and learning rate (s only)

**What:** s at 1280 and 30 epochs, with AdamW at lr0 0.0007 (half of auto's) and 0.003 (about
double), and YOLO26's MuSGD at 0.01. The baseline is `optimizer=auto` (AdamW at 0.001429). Each
run copies the momentum (0.9) and `warmup_bias_lr` (0) that auto sets, so only the optimizer and
lr0 change. A cosine schedule (`cos_lr`) was to follow only if an arm won.

**Why:** `auto` picks the optimizer and lr0 from a formula (class count and iteration count), not
from this data. YOLO26 also ships a new optimizer, MuSGD, which its paper credits with +0.4 AP.
The research behind the stage order ranked optimizer and lr0 third by expected effect, after image
size and epochs, at a small-to-medium effect.

```bash
uv run python -m mlops_cv.tuning.tune configs/tuning/yolo26-visdrone-vid/stage4.yaml
```

| Optimizer | lr0 | val mAP50-95 | person | vehicle | two-three-wheeler | mAP50-95_small | last six | best epoch |
|---|---|---|---|---|---|---|---|---|
| **AdamW (auto)** | **0.001429** | **30.8** | 26.5 | 42.9 | 22.9 | 13.5 | 30.6 | 25 |
| AdamW | 0.0007 | 30.7 | 28.6 | 41.3 | 22.0 | 13.4 | 30.0 | 22 |
| AdamW | 0.003 | 30.5 | 26.3 | 42.6 | 22.6 | 14.1 | 30.2 | 25 |
| MuSGD | 0.01 | 31.1 | 27.9 | 42.2 | 23.3 | 12.3 | 30.4 | 9 |

**Result:** `optimizer=auto` stays, so `cos_lr` never ran. The four runs span only 0.6 pt. MuSGD's
+0.3 is inside the margin and comes from an early `best.pt` (epoch 9).

## 7. Stage 5: zoom and mosaic (s only)

**What:** two augmentations, crossed on s at 1280 and 30 epochs:
- `scale` is a random resize: the default 0.5 means 0.5×–1.5×, and 0.25 means 0.75×–1.25×.
  `[0.75, 1.5]` never shrinks below 0.75× and enlarges up to 1.5×; it stands in for training on
  enlarged tiles.
- `mosaic` stitches four images into one: on every image (1.0, the default) or half of them (0.5).

**Why:** shrinking already-tiny objects can erase them. Mosaic is said to help small objects, yet
the usual small-dataset advice is to halve it.

```bash
uv run python -m mlops_cv.tuning.tune configs/tuning/yolo26-visdrone-vid/stage5.yaml
```

| scale | mosaic | val mAP50-95 | person | vehicle | two-three-wheeler | mAP50-95_small | last six | best epoch |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 1.0 | 30.8 | 26.5 | 42.9 | 22.9 | 13.5 | 30.6 | 25 |
| 0.25 | 1.0 | 30.4 | 26.7 | 42.4 | 22.0 | 13.7 | 29.6 | 20 |
| **[0.75, 1.5]** | **1.0** | **31.4** | 27.5 | 43.4 | 23.2 | 14.9 | 31.0 | 27 |
| 0.5 | 0.5 | 31.2 | 27.7 | 43.3 | 22.5 | 13.7 | 30.8 | 27 |
| 0.25 | 0.5 | 30.5 | 26.6 | 42.1 | 22.8 | 13.1 | 29.9 | 19 |
| [0.75, 1.5] | 0.5 | 31.0 | 28.1 | 42.8 | 22.1 | 14.3 | 30.2 | 22 |

**Result:** the zoom range `[0.75, 1.5]` wins on s by 0.6 pt, with the last-six mean agreeing
(+0.4) and small-object AP up 1.4. Halving mosaic gains 0.4 on its own (inside the margin) and
nothing on top of the zoom. A narrower `scale=0.25` loses. The zoom is later dropped, because it
doesn't help n or m (stage 9).

## 8. Stages 6–7: mosaic until the end, and class weights (s only)

**What:** on s's stage-5 winner (1280, 30 epochs, `scale=[0.75, 1.5]`):
- Stage 6 sets `close_mosaic=0`, which keeps mosaic on to the last epoch instead of turning it off
  for the last 10.
- Stage 7 sets `cls_pw` to 0.5 and 1.0. It weights the classification loss by inverse class
  frequency; at 1.0 the weights for person, vehicle and two-three-wheeler are 0.90/0.50/1.60.

**Why:** ultralytics turns mosaic off for the last 10 epochs so that training ends on whole
frames, and stage 6 checks whether that helps here. Stage 7 targets two-three-wheeler, the rarest
and weakest class.

```bash
uv run python -m mlops_cv.tuning.tune configs/tuning/yolo26-visdrone-vid/stage6.yaml
uv run python -m mlops_cv.tuning.tune configs/tuning/yolo26-visdrone-vid/stage7.yaml
```

| Setup | val mAP50-95 | person | vehicle | two-three-wheeler | mAP50-95_small | last six | best epoch |
|---|---|---|---|---|---|---|---|
| **stage 5 winner (`close_mosaic=10`, `cls_pw=0`)** | **31.4** | 27.5 | 43.4 | 23.2 | 14.9 | 31.0 | 27 |
| `close_mosaic=0` | 31.0 | 27.6 | 42.8 | 22.4 | 14.1 | 30.6 | 26 |
| `cls_pw=0.5` | 30.9 | 27.7 | 43.1 | 21.9 | 13.6 | 30.6 | 23 |
| `cls_pw=1.0` | 30.4 | 27.5 | 42.2 | 21.4 | 14.2 | 29.6 | 21 |

**Result:** both lose. `cls_pw` even lowers the class it upweights most: two-three-wheeler falls
1.3 and 1.8 pt. Two other balancing ideas were ruled out without a trial:
- **Oversampling the rare class:** 81% of train frames already hold a two-three-wheeler, so
  repeating them barely moves its share of boxes (16.8% → 17.2%).
- **Weighting every video equally:** videos hold 3–72 frames, and equal weighting would lower the
  rare class's share.

## 9. Confirmation: s's change on n and m

**What:** s's only win, `scale=[0.75, 1.5]`, on n and m at 1280 and 30 epochs. One more m trial
freezes the whole backbone (`freeze=11`), planned because m at 1280 won stage 2.

**Why:** stages 4–7 ran on s only.

```bash
uv run python -m mlops_cv.tuning.tune configs/tuning/yolo26-visdrone-vid/confirm.yaml
```

| Trial | val mAP50-95 | mAP50-95_small | last six | best epoch | GPU h |
|---|---|---|---|---|---|
| **n, stage 2** | **28.4** | 13.9 | 28.2 | 29 | 0.34 |
| n + zoom | 28.0 | 14.8 | 27.8 | 26 | 0.33 |
| **m, stage 2** | **32.9** | 15.9 | 31.6 | 19 | 0.96 |
| m + zoom | 32.6 | 16.1 | 31.3 | 17 | 0.97 |
| m + zoom + `freeze=11` | 31.3 | 15.9 | 31.0 | 13 | 0.63 |

**Result:** the zoom lowers n and m by 0.4 and 0.2 pt, though it still lifts small-object AP. A
frozen backbone loses 1.5 pt.

## Final recipe

One recipe for every size: the ultralytics defaults at imgsz 1280 and 30 epochs, with the batch
from the table above. The zoom averages 30.67 over the three sizes against 30.70 without it, so it
is dropped, and s's winner is `stage2-s1280`.

| Model | Trial run | val mAP50-95 | mAP50-95_small | p50 ms |
|---|---|---|---|---|
| yolo26n | `stage2-n1280` | 28.4 | 13.9 | 12.1 |
| yolo26s | `stage2-s1280` | 30.8 | 13.5 | 16.2 |
| yolo26m | `stage2-m1280` | 32.9 | 15.9 | 25.0 |

Stages 2 to 9 took 20.9 GPU-hours. The gain comes from inference resolution, so a model must be
served at the imgsz it was trained at. Run at 640, the 1280 models score n/s/m 21.0/25.4/28.0 on
val, below the models trained at 640 (21.0/26.2/28.6).

## 10. Other ways to run the same weights: the second head and SAHI

**What:** two ways to run each winner's weights, with no retraining. They are recorded beside the
trial and never pick a winner.
- **One-to-many head (`end2end=False`).** YOLO26 has two heads. The served one-to-one head outputs
  one box per object without NMS. The one-to-many head outputs several boxes per object and
  removes duplicates with NMS.
- **SAHI sliced inference.** Each frame is cut into imgsz tiles with 20% overlap. The model
  predicts every tile plus the full frame, and NMS (IoU 0.5) merges the results. The table reports
  it per source-frame width, since val holds 3840-, 2720- and ≤1904-px-wide frames and test is
  mostly 1360 px wide.

**Why:** the second head costs nothing to try. SAHI is the standard trick for small objects.

```bash
uv run python -m mlops_cv.tuning.sahi_eval stage2-n1280 stage5-s1280-zoom0.75-1.5 stage2-m1280
uv run python -m mlops_cv.tuning.sahi_eval --slice 640 stage2-n1280 stage5-s1280-zoom0.75-1.5 stage2-m1280
```

The first command runs `<trial>-o2m` and `<trial>-sahi`; `--slice 640` runs `<trial>-sahi640`. s
ran on its zoom trial, its winner at the time. `sahi` comes with the `tuning` uv group.

**Full frame** is SAHI's control: the same weights and one-to-one head on the whole frame with no
tiles, scored through the same path as SAHI (`model.predict`, then the validator's scoring). It
differs from the trial column, which comes from `model.val`, by 0.1–0.3 pt because the two paths
pad frames differently. `model.val` pads each batch of 8 frames to one shared shape plus half a
stride; `model.predict` pads one frame at a time to the next multiple of 32. The image then sits
differently on the network's 32-px grid, which moves some detections of tiny objects. So SAHI is
read against full frame, never against the trial column. s's trial column is
`stage5-s1280-zoom0.75-1.5` (31.4), not `stage2-s1280` (30.8).

| Model | one-to-one (trial) | one-to-many | full frame | SAHI | SAHI − full frame: 3840 / 2720 / ≤1904 px | SAHI p50 ms |
|---|---|---|---|---|---|---|
| yolo26n | 28.4 | **30.2** | 28.5 | 29.8 | +8.5 / +1.6 / +0.3 | 95 |
| yolo26s | 31.4 | **32.6** | 31.2 | 31.6 | +4.6 / −0.3 / +0.9 | 150 |
| yolo26m | 32.9 | **34.0** | 32.6 | 31.9 | +4.0 / −2.7 / +0.6 | 260 |

**Result:**
- **The one-to-many head scores 1.1–1.8 pt higher** on the same weights, more than every stage
  after image size gained. It is an input to the serving default, not a training change.
- **SAHI pays off only on 3840-wide frames.** At ≤1904 px, where test sits, it gains 0.3–0.9 pt
  for 8–10× the latency. Not adopted for serving.
- **Smaller tiles lose.** 640-px tiles score n/s/m 26.7/26.9/29.2, against 29.8/31.6/31.9 with
  1280 tiles, at twice the latency.

## 11. Sliced training

**What:** each winner retrained on train frames plus their 1280 tiles. Slicing turns 1238 frames
into 3300 extra tiles, and a box stays in a tile if at least a tenth of it is inside. Frames plus
tiles make an epoch 3.7× longer:
- `sliced` keeps 30 epochs, so 3.7× the training steps.
- `sliced-e8` uses 8 epochs, the baseline's step count, with `close_mosaic` and warmup scaled down
  to match.

Val stays full frames.

**Why:** the SAHI paper's gains came from training on tiles and then predicting on tiles.

```bash
uv run python -m mlops_cv.tuning.slices --imgsz 1280   # writes images/train-sliced1280 into the subset
export DATA__DATASET_YAML=configs/tuning/yolo26-visdrone-vid/datasets/VisDrone-VID-merged-sliced1280.yaml
uv run python -m mlops_cv.tuning.tune configs/tuning/yolo26-visdrone-vid/sliced.yaml
uv run python -m mlops_cv.tuning.tune configs/tuning/yolo26-visdrone-vid/sliced-e8.yaml
unset DATA__DATASET_YAML
uv run python -m mlops_cv.tuning.sahi_eval sliced-n1280 sliced-s1280-zoom0.75-1.5 sliced-m1280 \
  sliced-e8-n1280 sliced-e8-s1280-zoom0.75-1.5 sliced-e8-m1280
```

Each cell reads winner → `sliced` → `sliced-e8`.

| Model | SAHI | SAHI 3840 / 2720 / ≤1904 px | full frame | one-to-many |
|---|---|---|---|---|
| yolo26n | 29.8 → 29.4 → **30.7** | 15.9 / 33.7 / 30.9 → 14.5 / 35.6 / 31.4 (e8) | 28.5 → 28.2 → 28.4 | 30.2 → 29.7 → 29.9 |
| yolo26s | 31.6 → 30.5 → 31.4 | 15.1 / 36.0 / 32.5 → 19.6 / 36.3 / 31.7 (e8) | 31.2 → 30.2 → 30.0 | 32.6 → 31.4 → 31.7 |
| yolo26m | 31.9 → 31.3 → 31.7 | 19.8 / 35.6 / 33.8 → 21.9 / 35.3 / 33.1 (e8) | 32.6 → 31.5 → 31.5 | 34.0 → 33.5 → 32.9 |

**Result:**
- **At 30 epochs it overfits.** Every `sliced` run peaks at epoch 9–10 and loses on every read.
- **At equal steps, only n's SAHI score clears the margin** (+0.9). s and m tie on SAHI and lose
  1.1–1.2 pt on full frames.
- **Nothing changes.** n's best, 30.7 at 97 ms, is still below s's full-frame one-to-many score
  (32.6) at a fraction of the latency. The two stages took about 8 GPU-hours.

## 12. Test read

**What:** the final recipe's `best.pt` scored on test (337 frames, 17 videos) at 640 (stage 1)
and 1280 (stage 2), with the served one-to-one head, and at 1280 also with the one-to-many head.
Per-class and small-object columns are the one-to-one head. The s at 640 row is the mean of three
seeds (22.5/22.9/22.8), and latency is the trial's own.

**Why:** to check that val's picks hold on videos the selection never saw. It picks nothing.

```bash
uv run python -m mlops_cv.tuning.test_read stage1-n640 stage1-s640-seed0 stage1-s640-seed1 \
  stage1-s640-seed2 stage1-m640 stage2-n1280 stage2-s1280 stage2-m1280   # <trial>-test
uv run python -m mlops_cv.tuning.test_read --o2m stage2-n1280 stage2-s1280 stage2-m1280   # <trial>-test-o2m
```

| Model | imgsz | val mAP50-95 | test mAP50-95 | test one-to-many | person | vehicle | two-three-wheeler | test mAP50-95_small | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| yolo26n | 640 | 21.0 | 18.5 | – | 9.4 | 36.7 | 9.4 | 11.8 | 11.0 |
| yolo26s | 640 | 26.3 | 22.7 | – | 14.1 | 41.8 | 12.4 | 14.3 | 10.3 |
| yolo26m | 640 | 28.6 | 25.1 | – | 16.8 | 45.0 | 13.5 | 15.2 | 12.0 |
| yolo26n | 1280 | 28.4 | 25.0 | 26.2 | 17.4 | 44.5 | 13.1 | 14.7 | 12.1 |
| yolo26s | 1280 | 30.8 | 26.6 | 27.8 | 19.0 | 46.3 | 14.6 | 16.3 | 16.2 |
| **yolo26m** | **1280** | 32.9 | 28.3 | **29.5** | 20.7 | 48.1 | 16.1 | 18.4 | 25.0 |

**Result:**
- **Val's ranking holds.** 1280 beats 640 for every size, m > s > n, and n at 1280 still matches
  m at 640 (25.0 vs 25.1).
- **Test reads 2.5–4.6 pt below val.**
- **The one-to-many head gains 1.2 pt for every size**, as on val.
- **Seed noise on test is 0.34 pt** (s at 640), close to val's 0.25, so the margin holds.

## 13. Third-party comparison

**What:** dronefreak's public YOLO26 s and m at 1280, trained on VisDrone-DET (still images, ten
classes), scored on our test split beside our own s and m, with both heads.
[visdrone-yolo26s](https://huggingface.co/dronefreak/visdrone-yolo26s)@`dc042f4` and
[visdrone-yolo26m](https://huggingface.co/dronefreak/visdrone-yolo26m)@`cf0d3e8` are
AGPL-3.0, so they are evaluated here and never redistributed. Their ten classes are mapped onto
our three, and our models go through the same scoring path. That path reads our models 0.2–0.4 pt
below the test read.

**Why:** nobody publishes a YOLO26 model or score for VisDrone-VID, and every public VisDrone YOLO
checkpoint is DET-trained. DET shares VID's drones, categories and labelling, so this tests whether
a small tuned VID model beats the best public YOLO26 on our footage. It can't separate recipe from
training data.

```bash
uv run python -m mlops_cv.tuning.third_party
# downloads the pinned checkpoints into runs/third-party/;
# runs dronefreak-yolo26{s,m}1280-test[-o2m] and stage2-{s,m}1280-predict-test[-o2m]
```

The headline uses the 13 test videos that don't overlap VisDrone-DET (264 frames, without
`uav0000201_00000_v`, `uav0000297_00000_v` and both `uav0000249_*`). Different training data,
so this is context, not a like-for-like benchmark.

| Model | head | 13-video mAP50-95 | mAP50 | person | vehicle | two-three-wheeler | mAP50-95_small | 17-video mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| dronefreak s | one-to-one | 25.6 | 56.2 | 16.9 | 45.7 | 14.1 | 13.5 | 24.0 |
| ours s | one-to-one | 26.8 | 55.1 | 19.3 | 46.1 | 14.9 | 18.0 | 26.4 |
| dronefreak s | one-to-many | 27.1 | 58.9 | 19.0 | 47.0 | 15.5 | 14.4 | 25.7 |
| ours s | one-to-many | 27.9 | 57.1 | 20.6 | 47.4 | 15.7 | 18.3 | 27.6 |
| dronefreak m | one-to-one | 26.7 | 58.2 | 18.6 | 45.6 | 15.8 | 13.5 | 25.2 |
| ours m | one-to-one | 28.1 | 57.6 | 20.7 | 47.7 | 15.8 | 19.1 | 28.0 |
| dronefreak m | one-to-many | 28.0 | 60.2 | 20.2 | 46.8 | 16.9 | 14.3 | 26.6 |
| **ours m** | **one-to-many** | **29.1** | 59.0 | 21.8 | 49.4 | 16.0 | 19.3 | 29.1 |

**Result:**
- **Ours wins on the same head, for both sizes:** s by 1.2/0.8 pt and m by 1.4/1.1 pt
  (one-to-one/one-to-many). Any leakage from the unchecked DET images could only favour them.
- **They lead on mAP50** (+0.6–1.8 pt) but trail on mAP50-95: they find the objects but place the
  boxes less tightly. Small-object AP is 3.9–5.6 pt higher for ours.
- **Latency is the same** (same architecture).

Published numbers, for context only. Every row uses a different protocol, and none is directly
comparable. On the same weights, the official VisDrone toolkit reads 4.3–4.8 AP above pycocotools.

| Model | Data | Protocol | mAP50-95 |
|---|---|---|---|
| dronefreak YOLO26 s / m @1280 | DET test-dev, 10 classes | ultralytics val, 500 boxes | 26.43 / 29.24 |
| the same weights, read here | VID test, 17 videos, merged | this repo, one-to-many | 25.7 / 26.6 |
| XS-VID YOLOv8-S / YOLOv8-L / YOLOv9-C @1024 | VID test-dev or val (the paper says both) | COCO-style, unstated | 13.2 / 16.0 / 15.5 |
| arXiv 2605.22605 YOLOv8n / YOLOv8s / YOLO11n @640 | VID val | unstated | 10.4 / 13.6 / 12.2 |
| FGFA / D&T / FPN | VID test-dev | official toolkit | 14.44 / 14.21 / 12.93 |

## 14. Train frame stride

**What:** n at 1280 on the final recipe, trained on denser train splits: every 10th, 5th and 2nd
frame instead of every 20th. Val and test stay at every 20th frame, identical in every subset.

**Why:** the full dataset holds 20× more train frames than the subset. This checks whether more of
them help.

```bash
SUBSETS=<datasets>/subsets
for s in 10 5 2; do
  uv run python -m mlops_cv.pipelines.ingest_pipeline --runner DirectRunner \
    --train-frame-stride $s --output-dir $SUBSETS/visdrone-vid-train-s$s
  export DATA__SUBSET_DIR=$SUBSETS/visdrone-vid-train-s$s
  uv run python -m mlops_cv.pipelines.profile_pipeline --runner DirectRunner   # data version tag
  uv run python -m mlops_cv.tuning.tune configs/tuning/yolo26-visdrone-vid/stride$s.yaml
done
unset DATA__SUBSET_DIR
uv run python -m mlops_cv.tuning.test_read stride10-n1280
uv run python -m mlops_cv.tuning.test_read --o2m stride10-n1280
```

| Train stride | train frames | val mAP50-95 | person | vehicle | two-three-wheeler | mAP50-95_small | last six | best epoch | GPU h |
|---|---|---|---|---|---|---|---|---|---|
| 20 (`stage2-n1280`) | 1238 | 28.4 | 24.4 | 41.5 | 19.5 | 13.9 | 28.2 | 29 | 0.34 |
| **10** | 2446 | **29.4** | 25.5 | 41.4 | 21.2 | 14.1 | 28.7 | 22 | 0.67 |
| 5 | 4868 | 29.1 | 25.2 | 42.4 | 19.7 | 12.4 | 28.2 | 14 | 1.33 |
| 2 | 12112 | 29.1 | 24.3 | 42.1 | 21.0 | 12.7 | 26.6 | 8 | 3.27 |

**Result:**
- **Stride 10 wins on val**, 1.0 pt over stride 20. Going denser doesn't help further, so stride 1
  was not run.
- **Denser splits peak earlier and then decline.** The best epoch moves from 29 to 8. The extra
  frames are near-duplicates from the same 56 videos: more steps, not more variety.
- **Test doesn't confirm it.** Stride 10 reads 24.7 (one-to-many 25.8) against stride 20's 25.0
  (26.2), within test's seed noise.
- **Not adopted.** Stride 10 doubles training time for a gain test doesn't show, so the subset
  keeps stride 20. That is a cost call, not the win-margin rule.

## Summary

- **Recipe for every size:** the ultralytics defaults at imgsz 1280 and 30 epochs. Batch is 4/4/2
  for n/s/m on the 8 GB GPU, with `nbs=64`.
- **What mattered:**
  - Training on VisDrone: +7–9 pt over the pretrained models.
  - Image size 1280: +4.3–7.4 pt over 640. It was the only setting that won for every size.
- **What didn't help:**
  - more epochs (overfitting);
  - optimizer and learning rate;
  - the zoom augmentation (won on s only), mosaic and `close_mosaic`;
  - class-weighted loss;
  - a frozen backbone;
  - sliced training;
  - a denser train split (won on val, not confirmed on test, 2× the cost).
- **Inference:**
  - The one-to-many head adds 1.1–1.8 pt on val and 1.2 on test with the same weights.
  - SAHI helps only on 4K frames, at 8–10× the latency, so it is not adopted.
- **Final scores** (n/s/m):

  | | n | s | m |
  |---|---|---|---|
  | val | 28.4 | 30.8 | 32.9 |
  | test, one-to-one | 25.0 | 26.6 | 28.3 |
  | test, one-to-many | 26.2 | 27.8 | 29.5 |

- **Against public VisDrone checkpoints:** the recipe beats dronefreak's DET-trained YOLO26 s and m
  by 1.2 and 1.4 pt on the test videos they didn't train on.
- **Cost:** about 21 GPU-hours for stages 2–9, 8 for sliced training and 5 for the stride runs.
