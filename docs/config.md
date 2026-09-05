# Configuration

One code path runs in every environment; configuration comes from a single `.env` file plus the
OS environment, and only the values change. Local development reads `.env`; containers/cloud inject
variables or secrets.

## Precedence

Configuration is layered, **lowest precedence first**:

```
.env   <   OS environment
```

- **`.env`** — your local values (gitignored); `cp` it from the committed **`.env.example`**.
  Optional — the app runs on the model defaults without it.
- **OS environment** — overrides the file. In containers/cloud, inject variables directly
  (Cloud Run env vars / Secret Manager) and ship **no** `.env` file.

## Files

| File | Purpose |
|------|---------|
| `.env.example` | Template (git tracked) |
| `.env` | Your local copy / overrides — `cp .env.example .env` (optional; runs on model defaults without it) |

Local setup (optional — runs on defaults without it):

```bash
cp .env.example .env
```

## Usage

```python
from mlops_cv.config import get_settings

settings = get_settings()          # cached; reads .env + OS env (OS wins)
print(settings.log_level)
```

`load_settings(base_dir=…)` builds an uncached instance from `<base_dir>/.env` — used by tests to
isolate from the repo's `.env`.

## Nested settings

`Settings` sets `env_nested_delimiter="__"`, so a nested config group is set with the `__`
separator: `MLFLOW__TRACKING_URI=...` maps to `settings.mlflow.tracking_uri`.

## Quick check

```bash
uv run python -c "from mlops_cv.config import get_settings; print(get_settings().model_dump())"
```

Precedence is covered by [`tests/unit/test_settings.py`](../tests/unit/test_settings.py).
