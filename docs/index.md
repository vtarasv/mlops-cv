# Documentation

Per-component docs for the **mlops-cv** project. Each component adds or extends a
page here as it lands.

## Available now

- [Configuration & environment switch](config.md) — layered `.env` settings,
  `local`/`dev`/`stage`/`prod`, precedence rules.

## Planned

| Doc | Topic |
|-----|-------|
| `gpu-setup.md` | Blackwell/CUDA torch install, nvidia-container-toolkit on Fedora |
| `data.md` | VisDrone-VID download, frame/label conversion, 3-class merge |
| `mlflow.md` | MLflow + Postgres + MinIO Compose stack |
| `training.md` | YOLO26s transfer-learning recipe, MLflow logging |
| `evaluation.md` | Metrics (P/R/mAP), gate, benchmark harness |
| `orchestration.md` | Airflow continuous-training DAG |
| `batch-pipeline.md` | Apache Beam tiling pipeline |
| `streaming.md` | Redpanda/Kafka streaming inference |
| `optimization.md` | ONNX / TensorRT FP16 + INT8 (edge-sim) |
| `containers.md`, `serving.md` | Compose parity, FastAPI/Triton serving |
| `monitoring.md` | Drift detection + drift-triggered continuous training |
| `iac.md` | Terraform + GCP managed mirrors |
