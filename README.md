# mlops-cv — aerial object detection, MLOps level-2

Demo of a Google-style **[MLOps level-2](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)**
(continuous delivery + continuous training) pipeline for **aerial object detection from
video**. It trains [YOLO26s](https://github.com/ultralytics/ultralytics) (transfer learning
from COCO) on the [VisDrone-VID](https://docs.ultralytics.com/datasets/detect/visdrone)
benchmark and wires it through experiment tracking, orchestration, batch + streaming data
pipelines, model optimization, and a cloud tier.

> **Note:** Developed with AI assistance (Claude Code).

## Quickstart

```bash
# 1. Create the venv (Python 3.12) and install deps
make setup

# 2. Lint, format-check, test (what CI runs)
make ci

# 3. Inspect the active configuration
uv run python -c "from mlops_cv.config import get_settings; print(get_settings().model_dump())"
```

## Layout

```
src/mlops_cv/      # application code (config, data, training, …)
tests/unit/        # unit tests (CPU-only in CI; gpu/docker marks skipped)
docs/              # per-component docs
```
