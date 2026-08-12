# Documentation

Per-component docs for the **mlops-cv** project. Each component adds or extends a
page here as it lands.

## Available now

- [Configuration & environment switch](config.md) — layered `.env` settings,
  `local`/`dev`/`stage`/`prod`, precedence rules.
- [GPU setup](gpu-setup.md) — CUDA torch install, NVIDIA Container Toolkit.
- [VisDrone-VID dataset](data.md) — download, frame/label conversion, 3-class merge, validation.
- [MLflow tracking & registry](mlflow.md) — Postgres + RustFS + MLflow Compose stack.
- [Training](training.md) — YOLO26s transfer-learning recipe, MLflow logging.
- [Evaluation](evaluation.md) — metrics (P/R/mAP), promotion gate, demo videos, error analysis,
  latency stub.
- [Orchestration](orchestration.md) — Airflow continuous-training DAG (LocalExecutor,
  DockerOperator GPU tasks, asset-event trigger).
- [Batch data pipelines](batch-pipeline.md) — Apache Beam ingestion (raw → subset + demo store)
  and dataset profiling / drift baseline.
- [Streaming inference](streaming.md) — Redpanda/Kafka: demo-store producer, GPU champion
  consumer, anomaly alerts; delivery semantics + scaling paths.
- [Model optimization](optimization.md) — serving variants for two targets (GPU server: portable
  runtime + compiled FP16 engine; Raspberry Pi 5 edge: NCNN), every artifact published, the
  accuracy-vs-latency record + on-device benchmark harness.
- [Containers & Compose stacks](containers.md) — image inventory, the group-opt-in build
  pattern, stack topology, GPU policy.
- [Serving](serving.md) — the HTTP detection service: champion's published ONNX graph on
  onnxruntime CUDA, pre/post contract, API, observability.

## Planned

| Doc | Topic |
|-----|-------|
| `monitoring.md` | Drift detection + drift-triggered continuous training |
| `iac.md` | Terraform + GCP managed mirrors |
