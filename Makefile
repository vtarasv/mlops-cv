.PHONY: setup lint fmt test ci clean

# Create the venv (Python 3.12) and install core + dev dependency groups.
setup:
	uv venv --python 3.12
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

# What CI runs: lint + format-check + tests.
ci:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest -m "not gpu and not docker"

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
