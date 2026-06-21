# Configuration & environment switch

One code path runs in every environment; only the `ENV` variable — and the values it
pulls in — change. This is the single switch the whole project (local Docker Compose →
GCP) pivots on.

## Precedence

Configuration is layered, **lowest precedence first**:

```
.env   <   .env.<ENV>   <   OS environment
```

- **`.env`** — your local base (gitignored); `cp` it from the committed **`.env.example`**. Optional
  — the app runs on the model defaults without it.
- **`.env.<ENV>`** — per-environment overlay (gitignored); overrides `.env`.
- **OS environment** — overrides both. In containers/cloud, inject variables directly
  (Cloud Run env vars / Secret Manager) and ship **no** `.env.<ENV>` file.

`ENV` is one of `local` (default), `dev`, `stage`, `prod`.

## Files

| File | Purpose |
|------|---------|
| `.env.example` | Base template (git tracked) |
| `.env` | Your local copy / overrides — `cp .env.example .env` (optional; runs on model defaults without it) |
| `.env.local`, `.env.dev`, `.env.stage`, `.env.prod` | Per-env overlays — only the keys that differ; may hold secrets |

Local setup (optional — runs on defaults without it):

```bash
cp .env.example .env
```

## Usage

```python
from mlops_cv.config import get_settings

settings = get_settings()          # cached, uses the active ENV
print(settings.env, settings.log_level)
```

To load a specific environment explicitly (e.g. in scripts/tests):

```python
from mlops_cv.config import load_settings

prod = load_settings("prod")       # layers (.env, .env.prod), env pinned to "prod"
```

## Nested settings

`Settings` sets `env_nested_delimiter="__"`, so when a component adds a nested config group,
`MLFLOW__TRACKING_URI=...` will map to `settings.mlflow.tracking_uri`.

## Quick check

```bash
uv run python -c "from mlops_cv.config import get_settings; print(get_settings().model_dump())"
ENV=prod uv run python -c "from mlops_cv.config import get_settings; print(get_settings().env)"
```

Precedence is covered by [`tests/unit/test_settings.py`](../tests/unit/test_settings.py).
