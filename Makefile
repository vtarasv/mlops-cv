.PHONY: setup lint fmt test test-all ci clean gpu-smoke mlflow-up mlflow-down mlflow-logs train eval

MLFLOW_COMPOSE := docker compose -f docker-compose/docker-compose.mlflow.yml --env-file docker-compose/.env.mlflow

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

# Train YOLO26s on the VisDrone-VID subset, logging to MLflow.
train: mlflow-up
	uv run python -m mlops_cv.training.train

# Evaluate an MLflow model on the held-out test split -> metrics + report + gate in MLflow.
# Pass MODEL=<models:/aerial-object-detector@champion | runs:/<run_id>/weights/best.pt>.
eval: mlflow-up
	uv run python -m mlops_cv.eval.evaluate --model "$(MODEL)"

# What CI runs: lint + format-check + tests.
ci:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest -m "not gpu and not docker"

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
