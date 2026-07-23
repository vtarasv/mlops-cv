.PHONY: setup lint fmt test test-all ci clean gpu-smoke mlflow-up mlflow-down mlflow-logs \
	train eval ingest profile train-image beam-image airflow-up airflow-down airflow-logs airflow-env

MLFLOW_COMPOSE := docker compose -f docker-compose/docker-compose.mlflow.yml --env-file docker-compose/.env.mlflow
AIRFLOW_COMPOSE := docker compose -f docker-compose/docker-compose.airflow.yml --env-file docker-compose/.env.airflow

# Derived/exported wiring for the Airflow stack (fails loudly if any of these is missing):
#   TRAIN_IMAGE/BEAM_IMAGE : image tags — used for both `docker build` and the DAG's operators
#   HOST_RAW_DIR      : reuses DATA__RAW_DIR from the root .env (raw VisDrone-VID for the ingest task)
#   HOST_SUBSET_DIR   : reuses DATA__SUBSET_DIR from the root .env (the training subset the DAG mounts)
#   HOST_WEIGHTS      : reuses TRAINING__WEIGHTS from the root .env (pretrained-weights cache)
#   DOCKER_GID        : host docker group id — the scheduler needs it to use the mounted socket
export TRAIN_IMAGE := mlops-cv-train:0.1.0
export BEAM_IMAGE := mlops-cv-beam:0.1.0
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

# Auto-format with ruff.
fmt:
	uv run ruff format .

# Run the unit test suite (CPU-only; gpu/docker-marked tests skipped).
test:
	uv run pytest -m "not gpu and not docker"

# Run all tests, including GPU and docker-marked ones.
test-all:
	uv run pytest

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
	uv run python -m mlops_cv.training.train

# Evaluate an MLflow model -> test metrics + report + gate + demo videos + TP/FP crops in MLflow.
# Pass MODEL=<models:/aerial-object-detector@champion | runs:/<run_id>/weights/best.pt>.
# Add RUN_ID=<train run id> to log onto that run (one run per model version) instead of a new one.
eval: mlflow-up
	uv run python -m mlops_cv.eval.evaluate --model "$(MODEL)" $(if $(RUN_ID),--run-id "$(RUN_ID)")

# Build/refresh the training subset + demo store from raw dataset.
# Reads DATA__RAW_DIR, writes DATA__SUBSET_DIR (both from the root .env).
ingest:
	uv run python -m mlops_cv.pipelines.ingest_pipeline --runner DirectRunner

# Profile the subset: per-frame quality metrics + drift baseline -> <subset>/profile/.
profile:
	uv run python -m mlops_cv.pipelines.profile_pipeline --runner DirectRunner

# Build the GPU train/eval image the CT DAG's DockerOperator tasks run.
train-image:
	docker build -f docker/Dockerfile.train -t $(TRAIN_IMAGE) .

# Build the CPU Beam/data-prep image the CT DAG's build_subset + profile tasks run.
beam-image:
	docker build -f docker/Dockerfile.beam -t $(BEAM_IMAGE) .

# Print the derived host wiring the Airflow stack resolves ([brackets] surface stray whitespace).
airflow-env:
	@echo "TRAIN_IMAGE=[$(TRAIN_IMAGE)]"
	@echo "BEAM_IMAGE=[$(BEAM_IMAGE)]"
	@echo "HOST_RAW_DIR=[$(HOST_RAW_DIR)]"
	@echo "HOST_SUBSET_DIR=[$(HOST_SUBSET_DIR)]"
	@echo "HOST_WEIGHTS=[$(HOST_WEIGHTS)]"
	@echo "DOCKER_GID=[$(DOCKER_GID)]"

# Airflow CT stack (Postgres + api-server + scheduler + dag-processor + triggerer).
airflow-up: mlflow-up train-image beam-image airflow-env
	@test -n "$(HOST_SUBSET_DIR)" && mkdir -p "$(HOST_SUBSET_DIR)"
	$(AIRFLOW_COMPOSE) up -d --build --wait

airflow-down:
	$(AIRFLOW_COMPOSE) down

airflow-logs:
	$(AIRFLOW_COMPOSE) logs -f

# What CI runs: lint + format-check + tests.
ci:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest -m "not gpu and not docker"

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
