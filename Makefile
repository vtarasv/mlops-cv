.PHONY: setup lint fmt test ci clean gpu-smoke

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

# Real-hardware GPU check. Not run in CI.
gpu-smoke:
	uv run python scripts/gpu_smoke.py

# What CI runs: lint + format-check + tests.
ci:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest -m "not gpu and not docker"

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
