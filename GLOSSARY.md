# Aerial object detection

A continuously trained detector for objects in drone video: it learns from VisDrone-VID, serves
the current best model over HTTP and a live stream, and watches whether live footage still looks
like what that model learned from.

## Language

### Data

**Raw dataset**:
The VisDrone-VID benchmark as downloaded: drone video already cut into numbered frames, with
per-frame annotations in its ten raw categories.
_Avoid_: source data, VisDrone (unqualified)

**Sequence**:
One continuous drone recording in the raw dataset, stored as numbered frames, or one link session
on the stream. It is the unit that is split, keyed, replayed and baselined.
_Avoid_: video, scene, clip (except for a demo clip)

**Frame**:
One still image of a sequence, addressed by its sequence and frame index.
_Avoid_: image (when the position in a sequence matters)

**Raw category**:
One of VisDrone's ten original object labels, such as pedestrian, van or awning-tricycle.
_Avoid_: class (that word is reserved for merged classes)

**Merged class**:
One of the three classes the detector predicts: `person`, `vehicle`, `two-three-wheeler`. Each
raw category maps to exactly one merged class or is dropped. The classes are imbalanced, and
`two-three-wheeler` is the rare one.
_Avoid_: category, label

**Ignored region**:
An area VisDrone marks as too crowded or too low-resolution to score. Its boxes are dropped, not
painted out.
_Avoid_: masked region

**Subset**:
The training-ready dataset built from the raw dataset by keeping every n-th frame of every
sequence and merging classes. Training, evaluation, profiling and optimization all read it.
_Avoid_: dataset (unqualified), small dataset

**Frame stride**:
The n in "keep every n-th frame". It thins out the near-duplicate consecutive frames of a video.
_Avoid_: sampling rate, step

**Split**:
One of `train`, `val`, `test`. `val` selects the best weights; `test` is held out and read only
by evaluation and optimization.
_Avoid_: fold, test-dev (the raw name of the test split)

**Manifest**:
The subset's row-per-frame inventory. Its hash is the subset's identity.

**Demo clip**:
One of three fixed test sequences, picked so that each features a different merged class. They
keep qualitative comparisons stable across experiments.
_Avoid_: demo video (that is the rendered output)

**Demo store**:
The full-frame-rate copy of the demo clips, with ground truth, kept inside the subset. After
ingestion it is the only source of footage for demo videos and the stream.
_Avoid_: demo data, raw frames

### Dataset pipelines

**Ingestion**:
The pipeline that turns the raw dataset into the subset and its demo store. It is the only step
that reads the raw dataset.
_Avoid_: conversion, preprocessing

**Profiling**:
The pipeline that measures the subset (frame quality, box statistics, near-duplicates) and
produces the dataset profile and drift baseline.
_Avoid_: data validation (a separate pass/fail check)

**Dataset profile**:
The measured description of one subset: per-split and per-class distributions plus flagged frames.
_Avoid_: stats, quality report (that is just the flagged-frame part)

**Provenance stamp**:
The identity of a profile: the manifest hash plus the profiling parameters. A matching stamp
means the profile is current.
_Avoid_: checksum, cache key

**Data version**:
A dataset profile published to tracking under its provenance stamp. A training run names the
data version it learned from.
_Avoid_: dataset version, snapshot

**Drift baseline**:
Each training sequence's mean brightness, contrast and blur. Together these rows define what
"normal" footage looks like for a model.
_Avoid_: reference distribution, training stats

### Model lifecycle

**Continuous training**:
The scheduled or data-triggered loop that ensures the subset exists, profiles it, trains a
challenger, gates it, and optimizes and promotes it on a win.
_Avoid_: retraining, CT pipeline (as a synonym for the whole loop)

**New-data event**:
The signal that labeled data has been admitted to the dataset. It is the only data-driven
trigger for continuous training.
_Avoid_: data-changed webhook

**Model version**:
One registered set of trained weights, linked to the training run that produced it.
_Avoid_: model (when a specific version is meant), checkpoint

**Training run**:
The tracking run that produced a model version. Its test metrics and its record runs live there
too.
_Avoid_: experiment (in tracking, an experiment holds many runs)

**Champion**:
The model version that the champion alias currently points at, and the only one that is served
and monitored.
_Avoid_: production model, best model, latest model

**Challenger**:
A freshly trained model version under evaluation against the champion.
_Avoid_: candidate, new model

**Gate**:
The pass/fail decision on a challenger: every gate floor must hold *and* the challenger must
beat the champion's primary metric by the minimum improvement.
_Avoid_: quality check, validation

**Gate floor**:
An absolute lower bound on one test metric that a challenger must meet.
_Avoid_: threshold, minimum

**Primary metric**:
Test mAP50-95: the single number the champion/challenger comparison uses.
_Avoid_: score, accuracy

**Bootstrap win**:
A gate outcome when there is no champion yet: only the gate floors apply.
_Avoid_: first promotion, default pass

**Promotion**:
Pointing the champion alias at a challenger that passed the gate and is servable.
_Avoid_: deploy, release

**Servable**:
A model version that has the default serving variant of every deployment target published. Only
a servable model version can be promoted.
_Avoid_: deployable, ready

**Rollout**:
A service picking up a new champion. This happens only on restart, because every service
resolves the champion once at startup.
_Avoid_: hot reload, redeploy

**Record run**:
A child of a training run that collects everything one later activity learned about that model
version. There is one record run per activity (optimization, drift monitoring).
_Avoid_: ledger, evidence run, sub-run

### Evaluation

**Demo video**:
A demo clip rendered with ground truth and the model's detections overlaid, for side-by-side
visual comparison.
_Avoid_: demo, visualization

**Error-analysis crop**:
A crop of one low-confidence true positive or one high-confidence false positive. Its job is to
show where the model is unsure or confidently wrong.
_Avoid_: failure sample, hard example

### Tuning

**Tuning grid**:
The fixed, staged list of trials. Each stage varies one or two factors and carries its winner
into the next.
_Avoid_: sweep, search space, hyperparameter search

**Trial**:
One training run of one tuning-grid point. It is compared on val and never registered as a
model version.
_Avoid_: experiment (that holds many runs), training run (that produces a model version)

**Zero-shot baseline**:
A COCO-pretrained model scored without training, its COCO classes mapped onto the merged
classes. It shows what fine-tuning adds.
_Avoid_: out-of-the-box model, pretrained baseline

**Noise floor**:
The spread of val mAP50-95 across seeds of one trial setup. A trial must beat its stage's
winner so far by more than this to count as a win.
_Avoid_: variance, seed noise

### Optimization and serving

**Serving variant**:
One way of running the same weights: a specific runtime, numeric precision and input resolution.
_Avoid_: rung, export, flavour

**Variant ladder**:
The ordered set of serving variants. Each step changes one factor, so each step's cost or gain
can be read on its own.
_Avoid_: benchmark matrix

**Deployment target**:
A platform a serving variant is built for: the GPU **server**, or the Raspberry Pi 5 **edge**
device.
_Avoid_: backend, host

**Published graph**:
A model version's portable ONNX graph as published by optimization. Both the detection service
and the inference consumer serve the champion's.
_Avoid_: ONNX model, portable graph, exported model

**Engine fingerprint**:
A record of the hardware and compiler a compiled engine was built for. Anything that loads the
engine must match its own hardware against the fingerprint first.
_Avoid_: engine metadata

**Detection service**:
The HTTP service that answers one image with the champion's detections.
_Avoid_: API, inference server, model server

**Detection**:
One predicted box: a merged class, a confidence and a resolution-independent position.
_Avoid_: prediction, box (unqualified)

**Confidence floor**:
The minimum confidence for a detection to be reported. It has the same default on HTTP and on
the stream.
_Avoid_: confidence threshold, conf cutoff, operating point

### Streaming

**Producer**:
The process that publishes frames onto the stream, from the demo store or a video link,
optionally with a synthetic shift.
_Avoid_: camera, publisher

**Video link**:
A drone's or UGV's live encoded video feed, received on the ground.
_Avoid_: stream (that is the bus), feed (unqualified)

**Link session**:
One unbroken reception of a video link; a gap starts a new one. It is that link's sequence.
_Avoid_: connection

**Inference consumer**:
The stream reader that runs the published graph on each frame and emits a detection event.
_Avoid_: GPU worker, detector

**Anomaly consumer**:
The stream reader that applies the anomaly rule to detection events and emits alerts.
_Avoid_: alerter, watcher

**Frame address**:
A frame's identity on the stream: its sequence plus its frame index. Duplicates from replays
share it.
_Avoid_: message id, offset

**Frame time**:
The time the producer published a frame. Windows on the stream run on frame time, not
processing time.
_Avoid_: wall-clock, event time

**Detection event**:
The stream's answer for one frame: its detections and which model version produced them.
_Avoid_: result, inference output

**Anomaly rule**:
An alert condition on detection events: the windowed mean count of one merged class within one
sequence crossing a set level.
_Avoid_: alert rule, threshold rule

**Alert**:
The message the anomaly consumer emits when an anomaly episode opens.
_Avoid_: anomaly, notification

**Episode**:
A stretch during which a condition holds. It fires once when the condition begins and re-arms
only after the condition clears. Both the anomaly rule and drift monitoring work in episodes.
_Avoid_: incident, event

**Synthetic shift**:
A photometric distortion the producer can apply on purpose to exercise drift monitoring.
**Malignant** shifts (defocus) break the champion's detections; **benign** shifts (brightness)
leave them intact.
_Avoid_: corruption, augmentation, simulated drift

### Drift monitoring

**Drift monitor**:
The model-free process that judges whether live frames still look like the champion's drift
baseline.
_Avoid_: drift detector, data monitor

**Window**:
A fixed, non-overlapping batch of sampled frames that the drift monitor judges as one unit.
_Avoid_: batch, bucket

**Drift score**:
How far a window's mean of one statistic sits from the drift baseline's spread of training
sequence means, measured in that spread's standard deviations.
_Avoid_: z-score, distance, drift value

**Bar**:
The drift score that counts as drifted for one statistic: the farthest any training sequence
itself reaches, times a margin. It is derived at startup and never configured.
_Avoid_: threshold, limit, cutoff

**Drifted window**:
A window in which any single statistic's drift score exceeds its bar.
_Avoid_: anomalous window, alarm

**Episode evidence**:
The record of a drift episode's opening window: the crossing scores, the frames the window
covered, and its evidence frames. It is stored on the champion's drift record run.
_Avoid_: drift report, incident log

**Evidence frames**:
A few of an episode's opening-window frames kept as images, so the footage outlives the stream.
_Avoid_: snapshots, thumbnails

**Prediction health**:
Detection-rate, confidence and class-share readings shown next to drift scores. They give
context for a drift alarm but never trigger one.
_Avoid_: model drift, output drift
