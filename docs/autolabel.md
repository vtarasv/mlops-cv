# Auto-labeling: zero-shot labels for VisDrone-VID

These experiments measure how close a zero-shot auto-labeler gets to VisDrone-VID's ground truth.
An open-vocabulary detector finds boxes, a box filter cleans them up and a segmenter turns each
kept box into a mask. Nothing is trained. The answer tells a later effort whether auto-labels can
seed or replace human labels. Image mode labels single frames; sequence mode labels whole videos
and gives each object a track id. The [summary](#summary) at the end lists what was decided.
Scores are F1 at IoU 0.5, in %.

VisDrone-DET isn't used: on the same frames it labels 1.6× as many objects as VID, so a config
tuned on it would learn the wrong labeling protocol.

## How the experiments work

- **Data.** Four scoring sets, each with a COCO `ground_truth.json` that carries VID's `track_id`:

  | Set | Frames | Use |
  |---|---|---|
  | `train-sample` | 560: 10 evenly spaced from each of VID train's 56 videos | image-mode stages (nothing is trained) |
  | `val-stride20` | 146: every 20th VID val frame, the repo subset's val | image-mode finalists |
  | `val-full` | 2,846: every frame of VID val's 7 videos | sequence-mode stages |
  | `test-dev-full` | 6,635: every frame of VID test-dev's 17 videos | both labeling configs, read once at the end |

- **Labeling chain.** The detector is prompted with class names, each mapped to one of the 3
  merged classes:

  ```mermaid
  flowchart LR
      I[VID frame] --> D[detector] --> F[box filter] --> S[segmenter] --> R[COCO masks]
      F -- det boxes --> V[validator]
      S -- mask boxes --> V
  ```

- **Trial.** One labeling config. It logs to the MLflow experiment `aerial-auto-label`, with its
  auto-labels as `autolabels-<split>.json` (COCO, compressed RLE masks, `det_bbox` the detector's
  box).
- **Score.** Each trial scores two box sets through `score_with_ap_small` (ultralytics' validator)
  at the 3 merged classes: `det`, the filtered detector boxes, and `seg`, each mask's tight box.
  The F1 is read at the validator's max-F1 point. mAP and per-class scores are recorded but never
  decide.
- **Win margin: 0.5 pt.** A new setting replaces the current one only if it wins by 0.5 pt. A
  rerun gives the same score (nothing is trained), so there is no noise floor to measure.
- **A few settings at a time.** Each stage changes one or two settings, keeps the best and passes
  it on. Image-mode stages run on `train-sample`, sequence-mode stages on `val-full`.
- **Every box is scored.** The detector keeps its top 300 boxes above 0.001, so the validator's
  threshold sweep sees everything. The winner's max-F1 confidence becomes the **confidence
  floor**, which cuts only the stored auto-labels.
- **Stored detections.** A run over a whole set stores the detector's raw boxes in
  `<set>/detections/<hash of the detector settings>.npz`. A later trial with the same detector
  reads them and reruns only the filter and the segmenter.
- **Starting point.** Grounding DINO-T at a 0.25 floor, NMS 0.6, the keep-outer
  nested-box rule, a 0.4 score floor, then SAM 2.1 S.
- **Latency** is batch 1 per frame, decoded image to RLE, with the first 3 frames left out, on the
  laptop in [tuning](tuning.md#hardware).

### Ignored regions

**These scores don't compare with [tuning](tuning.md) or [evaluation](evaluation.md).** The
evaluator is the same, but the boxes fed to it differ:

| | Repo (training, tuning, evaluation) | Auto-labeling |
|---|---|---|
| Ground truth inside ignored regions | kept | dropped when ≥ half its area is inside |
| Predictions inside ignored regions | kept (hit is a false positive) | dropped by the same rule |

Ignored regions are crowds and distant traffic the annotators left unlabeled. A zero-shot
detector fires there, and the repo's convention would charge it false positives for real objects,
so the auto-labeler follows VisDrone's own toolkit and drops both sides.

## How to run

`.env` sets `DATA__RAW_DIR` (raw VisDrone-VID) and `DATA__AUTOLABEL_DIR` (the scoring sets);
see [data](data.md) for download instructions.

```bash
uv run python -m mlops_cv.autolabel.visdrone_vid   # builds the scoring sets
uv run --env-file .env python -m mlops_cv.autolabel configs/autolabel/detector.yaml --set train-sample
```

`--limit N` keeps the first N frames and `--sequence NAME` (repeatable) only those videos. A stage
file lists its trials, and `defaults` apply to every trial key by key:

```yaml
defaults:
  detector: {model: facebook/sam3, prompts: {pedestrian: person, car: vehicle, ...}, dtype: float16}
  box_filter: {min_side: 8, max_area: 0.05, nms_iou: 0.5}
trials:
  - {name: sam2.1-s, segmenter: facebook/sam2.1-hiera-small}
  - {name: own, segmenter: own}
```

Each section below gives the command that produced its table. Stage files are in
`configs/autolabel/`.

## 1. Box filter

**What:** Grounding DINO-T on the 10 raw VisDrone names, with 12 filters: NMS IoU {off, 0.5, 0.6,
0.7} × the nested-box rule {off, keep outer, keep inner}. The rule looks at two boxes where one
lies at least 80% inside the other and keeps either the outer or the inner one. Every filter also
drops boxes under 8 px on a side or over 5% of the frame, which removes almost no ground truth.
The incumbent is the starting point's filter (NMS 0.6, keep outer).

**Why:** the detector now keeps every box down to 0.001, so the filter alone decides what survives.

```bash
uv run python -m mlops_cv.autolabel configs/autolabel/box-filter.yaml --set train-sample
```

The detector returns 300 boxes per frame; the ground truth holds 49.

| NMS IoU | Nested-box rule | det F1 | P | R | Max-F1 conf | Kept per frame |
|---|---|---|---|---|---|---|
| off | off | 56.7 | 55.9 | 57.6 | 0.193 | 287 |
| off | keep outer | 39.1 | 50.2 | 32.0 | 0.142 | 88 |
| off | keep inner | 35.1 | 30.4 | 41.6 | 0.080 | 147 |
| **0.5** | **off** | **58.0** | 57.9 | 58.0 | 0.184 | 205 |
| 0.5 | keep outer | 45.2 | 56.6 | 37.6 | 0.165 | 93 |
| 0.5 | keep inner | 47.3 | 53.0 | 42.7 | 0.158 | 142 |
| 0.6 | off | 57.8 | 57.2 | 58.4 | 0.187 | 228 |
| 0.6 | keep outer (incumbent) | 43.8 | 56.2 | 35.9 | 0.161 | 91 |
| 0.6 | keep inner | 46.1 | 52.1 | 41.3 | 0.155 | 145 |
| 0.7 | off | 57.4 | 56.8 | 58.1 | 0.190 | 247 |
| 0.7 | keep outer | 42.0 | 54.2 | 34.3 | 0.155 | 89 |
| 0.7 | keep inner | 45.6 | 51.8 | 40.7 | 0.153 | 146 |

**Result:**
- **NMS 0.5 with no nested-box rule wins**, 14.2 pt over the incumbent.
- **The old floors were the bottleneck.** The incumbent scored 29.6 at its old floors (0.25 and
  0.4), 43.8 at 0.001, and dropping the nested-box rule takes it to 58.0.
- **The nested-box rule hurts both ways.** It ignores scores, so a low-scoring group box deletes
  the objects inside it (keep outer), and a low-scoring fragment deletes the object around it (keep
  inner).

## 2. Detector

**What:** stage 1's filter under Grounding DINO-T and -B, OWLv2-B/16, YOLOE-26-L-seg (at 640 and
1280) and SAM 3, each prompted with the 10 raw names and with plain nouns (person, car, van,
truck, bus, bicycle, motorcycle, tricycle).

**Why:** the detector sets the ceiling for everything after it.

```bash
uv run --env-file .env python -m mlops_cv.autolabel configs/autolabel/detector.yaml --set train-sample
```

The better prompt set per detector is in bold. Latency is the detector's p50, in fp32.

| Detector | Prompts | det F1 | P | R | Max-F1 conf | Detector ms |
|---|---|---|---|---|---|---|
| GDINO-T (incumbent) | raw | 58.0 | 57.9 | 58.0 | 0.184 | 232 |
| GDINO-T | **nouns** | 58.8 | 60.4 | 57.2 | 0.279 | 236 |
| GDINO-B | **raw** | 60.7 | 64.0 | 57.6 | 0.203 | 329 |
| GDINO-B | nouns | 58.0 | 60.1 | 56.0 | 0.254 | 327 |
| OWLv2-B/16 | raw | 50.9 | 50.4 | 51.5 | 0.274 | 201 |
| OWLv2-B/16 | **nouns** | 54.3 | 54.8 | 53.8 | 0.271 | 201 |
| YOLOE-26-L @ 640 | raw | 45.4 | 63.8 | 35.2 | 0.053 | 19 |
| YOLOE-26-L @ 640 | **nouns** | 52.1 | 60.7 | 45.6 | 0.124 | 19 |
| YOLOE-26-L @ 1280 | raw | 54.5 | 60.8 | 49.4 | 0.054 | 56 |
| YOLOE-26-L @ 1280 | **nouns** | 58.2 | 60.3 | 56.2 | 0.197 | 57 |
| **SAM 3** | **raw** | **66.5** | 68.0 | 65.0 | 0.597 | 1,600 |
| SAM 3 | nouns | 65.7 | 66.9 | 64.6 | 0.564 | 1,428 |

**Result:**
- **SAM 3 on the raw names wins**, 8.5 pt over the incumbent and 5.8 over GDINO-B, at 7× the
  latency. SAM 3 returns at most 200 boxes per prompt, but no frame holds more than 200 objects of
  one category, so the cap never binds.
- **Plain nouns help only the weaker detectors** (OWLv2 +3.4, YOLOE +3.7 to +6.7).
- **YOLOE needs the resolution:** +6.1 pt at 1280. No smaller model won, so OWLv2-L and YOLOE-26-X
  didn't run.

## 3. Prompts

**What:** SAM 3 with the 10 raw names against the 3 merged-class names (person, vehicle,
two-three-wheeler).

**Why:** 3 prompts mean 3 decoder passes instead of 10.

```bash
uv run --env-file .env python -m mlops_cv.autolabel configs/autolabel/prompts.yaml --set train-sample
```

| Prompts | det F1 | P | R | two-three-wheeler mAP50-95 | Detector ms |
|---|---|---|---|---|---|
| **10 raw names** | **66.5** | 68.0 | 65.0 | 18.2 | 1,600 |
| 3 merged names | 45.4 | 45.0 | 45.9 | 0.0 | 995 |

**Result:** the raw names stay, 21.1 pt ahead. SAM 3 doesn't know `two-three-wheeler`: the class
scores 0, while person and vehicle lose under 2 pt.

## 4. Tiling

**What:** SAM 3 on slices plus the full frame, at zoom 1.25 (806 px slices) and 2 (504 px). The
passes merge by dropping boxes cut by a slice edge, then NMS. One more row at 1.25 swaps the edge
rule for "the larger box wins". The incumbent is untiled.

**Why:** SAM 3 squashes every frame to 1008 px, so a 2720 px frame runs at 0.37× and its small
objects shrink further. SAHI-style slicing is the standard fix.

Tiling was too slow in fp32 (6 h), so **SAM 3 runs in fp16 from this stage on**. On the untiled
row, fp16 scores 66.3 against fp32's 66.5 and moves boxes 0.2 px at the median, at 484 ms instead
of 1,600.

```bash
uv run --env-file .env python -m mlops_cv.autolabel configs/autolabel/tiling.yaml --set train-sample
```

| Tiling | det F1 | P | R | mAP50-95 small | Max-F1 conf | Detector ms |
|---|---|---|---|---|---|---|
| **untiled (incumbent)** | **66.3** | 67.9 | 64.8 | 10.7 | 0.597 | 484 |
| zoom 1.25 | 65.5 | 64.7 | 66.2 | 9.0 | 0.737 | 3,389 |
| zoom 1.25, larger box wins | 44.8 | 53.8 | 38.4 | 7.3 | 0.674 | 3,389 |
| zoom 2 | 64.4 | 62.7 | 66.3 | 8.2 | 0.756 | 7,769 |

**Result:**
- **Untiled stays.** Slices add 1.5 pt of recall but cost 3–5 pt of precision: the extra boxes are
  confident false positives, and small-object AP falls.
- **The larger-box merge collapses**, because it merges boxes down to 0.001: most confident boxes
  get absorbed into a low-scoring or oversized box that the filter then drops.
- **Tiling costs 7–16×** SAM 3's untiled time.

## 5. Box filter recheck

**What:** stage 1's 12 filters again, on SAM 3 fp16.

**Why:** stage 1 picked the filter for Grounding DINO-T, whose boxes and scores differ.

```bash
uv run python -m mlops_cv.autolabel configs/autolabel/box-filter-recheck.yaml --set train-sample
```

| NMS IoU | Nested-box rule | det F1 | P | R | Max-F1 conf | Kept per frame |
|---|---|---|---|---|---|---|
| off | off | 65.1 | 66.1 | 64.1 | 0.626 | 274 |
| off | keep outer | 55.8 | 60.8 | 51.6 | 0.409 | 120 |
| off | keep inner | 49.0 | 60.2 | 41.4 | 0.517 | 138 |
| **0.5** | **off (incumbent)** | **66.3** | 67.9 | 64.8 | **0.597** | 151 |
| 0.5 | keep outer | 65.8 | 68.4 | 63.4 | 0.585 | 119 |
| 0.5 | keep inner | 57.8 | 64.6 | 52.3 | 0.585 | 130 |
| 0.6 | off | 66.5 | 67.8 | 65.3 | 0.597 | 167 |
| 0.6 | keep outer | 65.6 | 68.4 | 63.0 | 0.580 | 121 |
| 0.6 | keep inner | 56.9 | 64.4 | 50.9 | 0.582 | 135 |
| 0.7 | off | 66.7 | 67.7 | 65.6 | 0.598 | 184 |
| 0.7 | keep outer | 64.9 | 68.2 | 61.9 | 0.571 | 121 |
| 0.7 | keep inner | 55.9 | 64.2 | 49.5 | 0.579 | 137 |

**Result:** NMS 0.5 with no nested-box rule stays; 0.6 and 0.7 gain 0.2–0.3 pt, inside the
margin. Its max-F1 confidence, **0.597**, becomes the confidence floor. Keep outer now costs only
0.5 pt (against 12.8 on Grounding DINO-T), but keep inner still loses 8–11.

## 6. Segmenter

**What:** SAM 2.1 S (the incumbent) and L, and SAM 3's tracker, each prompted with the kept
boxes; and SAM 3's own masks, which come free with its boxes. Segmenters run in fp32. This stage
selects on `seg` F1.

**Why:** the mask is the label. Its tight box should fit the object at least as well as the
detector's box.

```bash
uv run --env-file .env python -m mlops_cv.autolabel configs/autolabel/segmenter.yaml --set train-sample
```

`det` F1 is 66.3 in every row. Latency is per frame, about 151 kept boxes.

| Segmenter | seg F1 | P | R | mAP50-95 small | Segmenter ms | Total ms |
|---|---|---|---|---|---|---|
| **SAM 2.1 S (incumbent)** | **66.4** | 68.0 | 65.0 | 11.6 | 629 | 1,111 |
| SAM 2.1 L | 66.2 | 67.8 | 64.8 | 11.2 | 841 | 1,323 |
| SAM 3 tracker | 65.3 | 67.1 | 63.6 | 10.5 | 1,321 | 1,807 |
| SAM 3 own masks | 64.4 | 65.7 | 63.2 | 10.2 | 285 | 766 |

**Result:** **SAM 2.1 S stays.** Its mask boxes score 0.1 pt above the detector's own boxes and lift
small-object AP from 10.7 to 11.6. L is 0.2 pt behind at 1.3× the time.

## 7. Segmenters on ground-truth boxes

**What:** each box-prompted segmenter given the ground-truth boxes instead of the detector's. Each
mask's tight box is compared with the box that prompted it. Its mask is also compared with
[AeroVIS](https://arxiv.org/abs/2607.08075)'s hand-reviewed masks, on the 460 `train-sample`
frames AeroVIS shares with VID. AeroVIS's masks were made by SAM 3 from these boxes, so the SAM 3
row is circular.

**Why:** to tell how much of the `seg` score is the detector and how much the masks.

```bash
uv run --env-file .env python -m mlops_cv.autolabel.gt_box_check --set train-sample \
  --aerovis <local AeroVIS folder>   # non-commercial
```

| Segmenter (GT boxes in) | Mean box IoU | F1 @ IoU 0.5 | F1 @ IoU 0.75 | Mask IoU vs AeroVIS |
|---|---|---|---|---|
| SAM 2.1 S | 76.4 | 97.7 | 60.2 | 83.4 |
| SAM 2.1 L | 75.6 | 97.0 | 56.9 | 83.7 |
| SAM 3 tracker | 75.5 | 97.1 | 56.2 | 89.2 (circular) |
| AeroVIS masks (reference) | 82.4 | 98.4 | 76.9 | — |

**Result:**
- **Given the right box, every segmenter keeps it at IoU 0.5** (97–98%), so the `seg` scores are
  bound by the detector, not the masks.
- **At IoU 0.75 they trail the reviewed masks by 17–21 pt.** Masks good enough for IoU 0.5 still
  need review for tight boxes.

## 8. Finalists on `val-stride20`

**What:** the winning image-mode config (`configs/autolabel/image-mode.yaml`) on 146 frames never
tuned on.

**Why:** to check that `train-sample`'s picks and floor hold on other videos.

```bash
uv run --env-file .env python -m mlops_cv.autolabel configs/autolabel/image-mode.yaml --set val-stride20
```

| Trial | Box | F1 | P | R | mAP50-95 | mAP50-95 small | Max-F1 conf | F1 at 0.597 | ms |
|---|---|---|---|---|---|---|---|---|---|
| **Image mode** | seg | **61.8** | 62.6 | 61.0 | 24.2 | 9.5 | 0.604 | 61.7 | 1,665 |

**Result:**
- **The floor transfers:** the max-F1 confidence here is 0.604, and cutting at 0.597 costs 0.1 pt.

## Sequence mode

A trial with a `tracker` section labels whole videos, so each object keeps one track id. Two kinds
of tracker:

- **Video trackers** (EdgeTAM, SAM 2.1 video) run the detector only on key frames, every K-th
  frame, and carry the masks between them. Key-frame boxes are matched one-to-one to the live
  tracks at IoU ≥ 0.3, and each box left over starts a new track. A track ends after 25 frames with an empty mask or 2 key frames without a
  match.
- **Box trackers** (BoT-SORT, ByteTrack: ultralytics' defaults) run the detector, filter and
  segmenter on every frame and link the boxes into tracks. Their `track_high_thresh` and
  `new_track_thresh` take the confidence floor, since their 0.25 default is YOLO's score scale.

The input is image mode's stored detections: SAM 3 fp16 boxes on the 10 raw names after the box filter. On each key frame, the boxes at or above the 0.597 floor seed new tracks or refresh live ones.

`seg` scores every frame's auto-labels; `det` scores the detector's boxes on key frames only.
Tracks are scored by TrackEval (`track/HOTA`, `IDF1`, `MOTA`, ID switches) per merged class.
Stages select on `seg` F1. Every trial reads one stored detector pass over `val-full`.

## 9. Key-frame interval

**What:** EdgeTAM at K = 1, 5, 10 and 20, beside image mode run on every frame
without tracking (reported only).

**Why:** fewer key frames are cheaper, but the masks drift for longer between them.

```bash
uv run --env-file .env python -m mlops_cv.autolabel configs/autolabel/key-interval.yaml --set val-full
uv run --env-file .env python -m mlops_cv.autolabel configs/autolabel/video-tracker-k1.yaml --set val-full
```

| Trial | seg F1 | det F1 | HOTA | IDF1 | ID switches | ms per frame |
|---|---|---|---|---|---|---|
| Image mode, no tracking (reference) | 61.6 | 61.2 | — | — | — | 1,724 |
| **K = 1** | **55.5** | 61.2 | 37.6 | 41.7 | 3,937 | 993 |
| K = 5 | 53.2 | 61.0 | 39.9 | 46.0 | 1,307 | 615 |
| K = 10 | 52.2 | 61.0 | 40.8 | 48.2 | 922 | 579 |
| K = 20 | 51.5 | 61.1 | 40.8 | 48.5 | 666 | 565 |

**Result:**
- **K = 1 wins**, 2.3 pt over K = 5, but loses on tracks: 3× the ID switches, since a track ends
  after 2 missed key frames, now 2 frames instead of 10. The tracker-model stage ran at K = 5.
- **EdgeTAM costs 6–10 pt against labeling every frame**, on every video: its masks between key
  frames fit worse than SAM 2.1 S's on fresh boxes.
- **The 4K video (3840×2160) is floor-bound.** The detector reaches 43.9 there, but SAM 3 scores
  distant objects below 0.597, so the cut auto-labels keep 17.9% recall and score 29.7.

## 10. Tracker model

**What:** at K = 5, EdgeTAM (the incumbent) against SAM 2.1 S video, and the box trackers BoT-SORT
and ByteTrack on SAM 2.1 S masks.

**Why:** EdgeTAM lost 8 pt to image mode. A stronger video tracker, or a box tracker on image
mode's labels, may keep the quality and add ids.

```bash
# expandable segments: SAM 2.1 S video may run out of memory
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --env-file .env \
  python -m mlops_cv.autolabel configs/autolabel/tracker-model.yaml --set val-full
```

| Trial | seg F1 | P | R | HOTA | IDF1 | MOTA | ID switches | ms per frame |
|---|---|---|---|---|---|---|---|---|
| **BoT-SORT + SAM 2.1 S** | **61.3** | 61.9 | 60.8 | 47.8 | 60.7 | 27.1 | 394 | 1,766 |
| ByteTrack + SAM 2.1 S | 61.0 | 64.1 | 58.3 | 44.1 | 53.6 | 28.4 | 1,481 | 1,742 |
| SAM 2.1 S video | 56.4 | 55.9 | 56.9 | 44.8 | 52.3 | 14.0 | 521 | 1,609 |
| EdgeTAM video (incumbent) | 53.2 | 53.3 | 53.2 | 39.9 | 46.0 | 5.0 | 1,307 | 615 |

**Result:**
- **BoT-SORT wins**, 8.1 pt over EdgeTAM and within 1.4 pt of image mode on every video.
  ByteTrack is 0.3 pt behind but loses 3.7 pt HOTA and switches ids almost 4× as often.
- **SAM 2.1 S video recovers 3.2 pt over EdgeTAM**, but its masks still drift between key frames.
- **No latency win.** BoT-SORT runs the detector and segmenter on every frame, as image mode does.

## 11. Confidence floor

**What:** BoT-SORT at floors 0.497, 0.597 (the incumbent) and 0.697.

**Why:** the floor was measured on `train-sample` frames, and the 4K video wants a lower one.

```bash
uv run --env-file .env python -m mlops_cv.autolabel configs/autolabel/confidence-floor.yaml --set val-full
```

| Floor | seg F1 | P | R | F1 at the floor | HOTA | IDF1 | ID switches |
|---|---|---|---|---|---|---|---|
| 0.497 | 61.7 | 62.9 | 60.6 | 60.9 | 47.5 | 59.2 | 527 |
| **0.597** | **61.3** | 61.9 | 60.8 | 61.3 | 47.8 | 60.7 | 394 |
| 0.697 | 58.3 | 65.9 | 52.2 | 58.3 | 45.1 | 57.6 | 405 |

`seg` F1 per video, each at its own max-F1 point:

| Video | det (uncut) | 0.497 | 0.597 | 0.697 |
|---|---|---|---|---|
| uav0000086_00000_v | 76.3 | 76.3 | 73.8 | 66.6 |
| uav0000117_02622_v | 65.3 | 65.9 | 65.5 | 57.8 |
| uav0000137_00458_v | 68.0 | 67.5 | 67.0 | 65.9 |
| uav0000182_00000_v | 50.4 | 51.9 | 51.4 | 50.2 |
| **uav0000268_05773_v** (3840×2160) | 43.9 | **42.6** | 29.2 | 17.5 |
| uav0000305_00000_v | 51.9 | 54.0 | 53.3 | 51.5 |
| uav0000339_00001_v | 68.2 | 68.2 | 67.4 | 64.9 |

**Result:**
- **0.597 stays.** 0.497 gains 0.4 pt, inside the margin, and only because the validator cuts it
  back at 0.604; at its own floor it scores 60.9.
- **The 4K video wants 0.497:** +13.4 pt, with every other video level or better. The set's one
  threshold follows the dense videos, and a floor per resolution is outside the grid.
- Sequence mode's labeling config is `configs/autolabel/sequence-mode.yaml`.

## 12. Test-dev read

**What:** both labeling configs, read once on `test-dev-full`: 6,635 frames from 17 videos, 680 to
2720 px wide, with no 4K video. Image mode runs first and stores the detector pass; sequence mode
reads it. Nothing selects.

**Why:** to check that the picks made on `train-sample` and `val-full` hold on videos never tuned
on.

```bash
uv run --env-file .env python -m mlops_cv.autolabel.visdrone_vid --set test-dev-full
uv run --env-file .env python -m mlops_cv.autolabel configs/autolabel/image-mode.yaml --set test-dev-full
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --env-file .env \
  python -m mlops_cv.autolabel configs/autolabel/sequence-mode.yaml --set test-dev-full
```

Mean latency per frame.

| Config | Set | seg F1 | P | R | mAP50-95 | HOTA | IDF1 | ID switches | ms per frame |
|---|---|---|---|---|---|---|---|---|---|
| Image mode | `val-full` | 61.6 | 62.7 | 60.5 | 24.3 | — | — | — | 1,724 |
| **Image mode** | `test-dev-full` | **61.5** | 64.2 | 59.0 | 22.5 | — | — | — | 1,183 |
| Sequence mode | `val-full` | 61.3 | 61.9 | 60.8 | 21.6 | 47.8 | 60.7 | 394 | 1,766 |
| **Sequence mode** | `test-dev-full` | **61.4** | 65.9 | 57.6 | 20.2 | 46.7 | 60.1 | 1,299 | 1,172 |

`seg` F1 per video, each at its own max-F1 point:

| Video | Size | Frames | det (uncut) | Image mode | Sequence mode |
|---|---|---|---|---|---|
| uav0000009_03358_v | 1360×765 | 219 | 61.8 | 62.2 | 63.0 |
| uav0000073_00600_v | 1920×1080 | 328 | 55.6 | 55.8 | 54.9 |
| uav0000073_04464_v | 1920×1080 | 312 | 61.7 | 61.9 | 60.9 |
| uav0000077_00720_v | 1360×765 | 780 | 68.5 | 67.9 | 68.2 |
| uav0000088_00290_v | 960×540 | 296 | 66.7 | 63.3 | 62.4 |
| uav0000119_02301_v | 1360×765 | 179 | 60.0 | 59.0 | 59.3 |
| uav0000120_04775_v | 1360×765 | 1000 | 64.9 | 64.2 | 63.7 |
| uav0000161_00000_v | 960×540 | 308 | 65.2 | 64.0 | 63.2 |
| uav0000188_00000_v | 960×540 | 260 | 42.5 | 43.6 | 43.3 |
| uav0000201_00000_v | 1344×756 | 677 | 38.3 | 40.9 | 40.5 |
| uav0000249_00001_v | 1920×1080 | 360 | 65.0 | 67.1 | 66.7 |
| uav0000249_02688_v | 1344×756 | 244 | 58.7 | 58.6 | 58.8 |
| uav0000297_00000_v | 1904×1071 | 146 | 49.8 | 50.3 | 50.3 |
| uav0000297_02761_v | 1904×1071 | 373 | 67.9 | 66.8 | 66.3 |
| uav0000306_00230_v | 680×382 | 420 | 50.6 | 49.9 | 49.7 |
| uav0000355_00001_v | 1360×765 | 468 | 61.6 | 62.1 | 61.8 |
| uav0000370_00001_v | 2720×1530 | 265 | 67.1 | 68.7 | 69.0 |

**Result:**
- **Val's reads hold.** Image mode scores 61.5 on test-dev against 61.6 on `val-full` and 61.8 on
  `val-stride20`; sequence mode scores 61.4 against 61.3. `train-sample`, where the stages
  selected, reads 5 pt higher (66.4).
- **The floor transfers.** Image mode's max-F1 confidence is 0.579, and cutting at 0.597 costs
  nothing (61.5).
- **Tracking adds ids at no F1 cost.** Sequence mode is within 1 pt of image mode on every video.
  HOTA and IDF1 sit about 1 pt under val's, but ID switches per frame rise 1.4×.
- **The videos spread from 41 to 69.** Every video is under 4K, so the floor problem of val's 4K
  video doesn't arise here.
- **Cost.** Test-dev's frames are smaller than val's: 1.2 s per frame. Image mode took 2.2 h, and
  sequence mode took 1.2 h on its stored pass.

## Viewing a trial

The annotation editor is FiftyOne 1.22.1 (`docker-compose/docker-compose.fiftyone.yml`), with raw
VID and the scoring sets mounted read-only at their host paths.

```bash
make fiftyone-up                        # http://localhost:5151
make autolabel-view RUN=<MLflow run id> # a run in aerial-auto-label
```

- Each scoring set is one dataset, `visdrone-vid-<set>`, with `ground_truth` and `ignored`. A trial
  adds `<run>_seg` (masks) and `<run>_det` (the detector's boxes); reloading overwrites them.
- Each field gets FiftyOne's own TP/FP/FN flags at IoU 0.5, for browsing only: they count every
  confidence, so the scores above come from the validator alone.
- Loading takes minutes: 2.3 min for one `train-sample` trial, 14 min for a sequence-mode trial on
  `val-full`.

## Summary

- **Image-mode config:** SAM 3 fp16 on the 10 raw VisDrone names, untiled → box filter (min side
  8 px, ≤ 5% of the frame, NMS 0.5) → SAM 2.1 S, auto-labels cut at confidence 0.597. `seg` F1
  66.4 on `train-sample`, 61.8 on `val-stride20` and 61.5 on `test-dev-full`. 1.2–1.7 s per frame.
- **Sequence-mode config:** the same chain on every frame, tracked by BoT-SORT. `seg` F1 61.3 and
  HOTA 47.8 on `val-full`; 61.4 and 46.7 on `test-dev-full`.
- **What mattered:**
  - Scoring every box and dropping the nested-box rule: 29.6 → 58.0 on Grounding DINO-T.
  - SAM 3 as the detector: +8.5 pt.
  - A box tracker over video trackers: +8.1 pt.
- **What didn't help:** plain nouns or merged-class names, tiling, a larger segmenter, SAM 3's
  own masks, video trackers (EdgeTAM, SAM 2.1 video).
- **Masks are not the bottleneck.** On ground-truth boxes they keep 97–98% at IoU 0.5, but trail
  hand-reviewed masks by 17–21 pt at IoU 0.75.
- **Not adopted:** a lower floor for 4K frames. It gains 13.4 pt on val's one 4K video, and
  test-dev has none.
