# Documentation

Per-component docs for the **mlops-cv** project. Each component adds or extends a
page here as it lands.

## Available now

- [Configuration & environment switch](config.md) — layered `.env` settings,
  `local`/`dev`/`stage`/`prod`, precedence rules.
- [GPU setup](gpu-setup.md) — CUDA torch install, NVIDIA Container Toolkit.
- [VisDrone-VID dataset](data.md) — download, frame/label conversion, 3-class merge, validation.
- [MLflow tracking & registry](mlflow.md) — Postgres + RustFS + MLflow Compose stack.
- [Training](training.md) — YOLO26s transfer-learning recipe, MLflow logging, demo videos.

## Planned

| Doc | Topic |
|-----|-------|
| `evaluation.md` | Metrics (P/R/mAP), gate, benchmark harness |
| `orchestration.md` | Airflow continuous-training DAG |
| `batch-pipeline.md` | Apache Beam tiling pipeline |
| `streaming.md` | Redpanda/Kafka streaming inference |
| `optimization.md` | ONNX / TensorRT FP16 + INT8 (edge-sim) |
| `containers.md`, `serving.md` | Compose parity, FastAPI/Triton serving |
| `monitoring.md` | Drift detection + drift-triggered continuous training |
| `iac.md` | Terraform + GCP managed mirrors |
