.PHONY: setup lint fmt typecheck test test-all ci clean gpu-smoke \
	train eval ingest profile optimize serve \
	beam-image train-image optimize-image serve-image stream-image monitor-image \
    mlflow-up mlflow-down mlflow-logs \
	airflow-up airflow-down airflow-logs airflow-env \
	streaming-up streaming-down streaming-logs \
	serving-up serving-down serving-logs \
	stack-up stack-down

MLFLOW_COMPOSE := docker compose -f docker-compose/docker-compose.mlflow.yml --env-file docker-compose/.env.mlflow
AIRFLOW_COMPOSE := docker compose -f docker-compose/docker-compose.airflow.yml --env-file docker-compose/.env.airflow
STREAMING_COMPOSE := docker compose -f docker-compose/docker-compose.streaming.yml --env-file docker-compose/.env.streaming
SERVING_COMPOSE := docker compose -f docker-compose/docker-compose.serving.yml --env-file docker-compose/.env.serving

# Derived/exported wiring for the Airflow + serving stacks (fails loudly if any is missing):
#   *_IMAGE           : image tags — used for `docker build`, the DAG's operators, and compose
#   HOST_RAW_DIR      : reuses DATA__RAW_DIR from the root .env (raw VisDrone-VID for the ingest task)
#   HOST_SUBSET_DIR   : reuses DATA__SUBSET_DIR from the root .env (the training subset the DAG mounts)
#   HOST_WEIGHTS      : reuses TRAINING__WEIGHTS from the root .env (pretrained-weights cache)
#   DOCKER_GID        : host docker group id — the scheduler needs it to use the mounted socket
export BEAM_IMAGE := mlops-cv-beam:0.1.0
export TRAIN_IMAGE := mlops-cv-train:0.1.0
export OPTIMIZE_IMAGE := mlops-cv-optimize:0.1.0
export STREAM_IMAGE := mlops-cv-stream:0.1.0
export SERVE_IMAGE := mlops-cv-serve:0.1.0
export MONITOR_IMAGE := mlops-cv-monitor:0.1.0
export HOST_RAW_DIR := $(shell sed -n 's/^DATA__RAW_DIR=//p' .env 2>/dev/null)
export HOST_SUBSET_DIR := $(shell sed -n 's/^DATA__SUBSET_DIR=//p' .env 2>/dev/null)
export HOST_WEIGHTS := $(shell sed -n 's/^TRAINING__WEIGHTS=//p' .env 2>/dev/null)
export DOCKER_GID := $(shell getent group docker | cut -d: -f3)

# Create the venv (from .python-version) and install all default groups.
setup:
	uv sync

# Lint with ruff.
lint:
	uv run ruff check .

# Auto-fix lint findings, then format.
fmt:
	uv run ruff check --fix .
	uv run ruff format .

# Static type check of the package.
typecheck:
	uv run pyright

# Run the unit test suite (CPU-only; gpu/docker-marked tests skipped).
test:
	uv run pytest -m "not gpu and not docker"

# Run all tests, including GPU and docker-marked ones.
test-all:
	uv run pytest

# What CI runs: lint + format-check + typecheck + tests.
ci:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run pytest -m "not gpu and not docker"

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__

# Real-hardware GPU check. Not run in CI.
gpu-smoke:
	uv run python scripts/gpu_smoke.py

# MLflow tracking + registry stack (Postgres + RustFS + MLflow server).
mlflow-up:
	$(MLFLOW_COMPOSE) up -d --build --wait

mlflow-down:
	$(MLFLOW_COMPOSE) down

mlflow-logs:
	$(MLFLOW_COMPOSE) logs -f

# Train YOLO26s on the train subset -> MLflow run + registered version.
train: mlflow-up
	uv run python -m mlops_cv.training

# Evaluate an MLflow model -> test metrics + report + gate + demo videos + TP/FP crops in MLflow.
# Pass MODEL=<models:/aerial-object-detector@champion | runs:/<run_id>/weights/best.pt>.
# Add RUN_ID=<train run id> to log onto that run (one run per model version) instead of a new one.
# Add PROMOTE=1 to set the champion alias on a gate pass (the DAG promotes from its own task).
eval: mlflow-up
	uv run python -m mlops_cv.evaluation --model "$(MODEL)" $(if $(RUN_ID),--run-id "$(RUN_ID)") \
		$(if $(PROMOTE),--promote)

# Build a model's serving variants + benchmark them -> optimization report in MLflow.
# Defaults to the champion; pass MODEL=<uri> for another, RUN_ID=<train run id> to record on a
# CHILD of that run (re-runs reuse it), VARIANTS=<a,b> to measure a subset of the ladder.
optimize: mlflow-up
	uv run python -m mlops_cv.optimize $(if $(MODEL),--model "$(MODEL)") \
		$(if $(RUN_ID),--run-id "$(RUN_ID)") $(if $(VARIANTS),--variants "$(VARIANTS)")

# onnxruntime-gpu finds the CUDA runtime through the system loader; in the serving container
# that is the CUDA base image, on the host it is the venv's nvidia wheels (torch's).
VENV_CUDA_LIBS := $(shell echo .venv/lib/python*/site-packages/nvidia/*/lib | tr ' ' ':')

# Serve the champion's published ONNX graph over HTTP (GPU; POST an image to /predict).
# Pass PORT=<n> to listen elsewhere. Needs a champion whose version carries the graph tag.
serve: mlflow-up
	LD_LIBRARY_PATH="$(VENV_CUDA_LIBS):$$LD_LIBRARY_PATH" \
		uv run python -m mlops_cv.serving $(if $(PORT),--port "$(PORT)")

# Build/refresh the training subset + demo store from raw dataset.
# Reads DATA__RAW_DIR, writes DATA__SUBSET_DIR (both from the root .env).
ingest:
	uv run python -m mlops_cv.pipelines.ingest_pipeline --runner DirectRunner

# Profile the subset: per-frame quality metrics + drift baseline -> <subset>/profile/, published to MLflow.
profile: mlflow-up
	uv run python -m mlops_cv.pipelines.profile_pipeline --runner DirectRunner

# Build the CPU Beam/data-prep image the CT DAG's build_subset + profile tasks run.
beam-image:
	docker build -f docker/Dockerfile.beam -t $(BEAM_IMAGE) .

# Build the GPU train/eval image the CT DAG's DockerOperator tasks run.
train-image:
	docker build -f docker/Dockerfile.cudnn-devel --target train -t $(TRAIN_IMAGE) .

# Build the GPU serving-variant image the CT DAG's optimize task runs.
optimize-image:
	docker build -f docker/Dockerfile.cudnn-devel --target optimize -t $(OPTIMIZE_IMAGE) .

# Print the derived host wiring the Airflow stack resolves ([brackets] surface stray whitespace).
airflow-env:
	@echo "BEAM_IMAGE=[$(BEAM_IMAGE)]"
	@echo "TRAIN_IMAGE=[$(TRAIN_IMAGE)]"
	@echo "OPTIMIZE_IMAGE=[$(OPTIMIZE_IMAGE)]"
	@echo "HOST_RAW_DIR=[$(HOST_RAW_DIR)]"
	@echo "HOST_SUBSET_DIR=[$(HOST_SUBSET_DIR)]"
	@echo "HOST_WEIGHTS=[$(HOST_WEIGHTS)]"
	@echo "DOCKER_GID=[$(DOCKER_GID)]"

# Airflow CT stack (Postgres + api-server + scheduler + dag-processor + triggerer).
airflow-up: mlflow-up beam-image train-image optimize-image airflow-env
	@test -n "$(HOST_SUBSET_DIR)" && mkdir -p "$(HOST_SUBSET_DIR)"
	@test -n "$(HOST_WEIGHTS)" && mkdir -p "$$(dirname "$(HOST_WEIGHTS)")"
	$(AIRFLOW_COMPOSE) up -d --build --wait

airflow-down:
	$(AIRFLOW_COMPOSE) down

airflow-logs:
	$(AIRFLOW_COMPOSE) logs -f

# Streaming stack (Redpanda broker + Console + one-shot topic init).
streaming-up: mlflow-up
	$(STREAMING_COMPOSE) up -d --wait

streaming-down:
	$(STREAMING_COMPOSE) down

streaming-logs:
	$(STREAMING_COMPOSE) logs -f

# Build the GPU serving image the serving stack runs.
serve-image:
	docker build -f docker/Dockerfile.cudnn-runtime --target serve -t $(SERVE_IMAGE) .

# Build the streaming-consumers image (all consumers run it).
stream-image:
	docker build -f docker/Dockerfile.cudnn-runtime --target stream -t $(STREAM_IMAGE) .

# Build the slim CPU drift-monitor image the serving stack runs.
monitor-image:
	docker build -f docker/Dockerfile.monitor -t $(MONITOR_IMAGE) .

# Serving stack: the detection service + both streaming consumers + the drift monitor
# + observability (Prometheus + Grafana).
serving-up: streaming-up serve-image stream-image monitor-image
	$(SERVING_COMPOSE) up -d --wait

serving-down:
	$(SERVING_COMPOSE) down

serving-logs:
	$(SERVING_COMPOSE) logs -f

# Everything at once: MLflow + CT images + Airflow + streaming + serving.
stack-up: airflow-up streaming-up serving-up

# Tear the whole stack down.
stack-down:
	-$(SERVING_COMPOSE) down
	-$(STREAMING_COMPOSE) down
	-$(AIRFLOW_COMPOSE) down
	$(MLFLOW_COMPOSE) down
