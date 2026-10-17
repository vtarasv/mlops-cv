# AGENTS.md

Local-first MLOps level-2 repo (continuous training + delivery) for aerial object detection from
video: VisDrone-VID → YOLO26s → MLflow → Airflow CT DAG → Beam data pipelines → Redpanda streaming
→ ONNX / TensorRT / NCNN variants → FastAPI serving + drift monitor, each a Docker Compose stack.

`docs/index.md` maps the components. Read the matching `docs/<component>.md` before changing one.

## Tooling

- Essential plugins: `mattpocock-skills`, `ponytail`, `context7`. Claude Code installs them from
  `.claude/settings.json` when you trust the folder. Other assistants: install the skills from
  `github.com/mattpocock/skills` and `github.com/DietrichGebert/ponytail`, plus the Context7 MCP
  server.
- Fetch docs for a package or tool with `context7:context7` before writing code against its API.
- The Makefile is the command reference. `make setup` syncs every default uv group.

## Agent skills

This is setup for `mattpocock-skills`.

### Issue tracker

Local markdown under `.scratch/<feature>/` (gitignored, machine-local). See
`docs/agents/issue-tracker.md`.

### Triage labels

The five default roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`,
`wontfix`), recorded as `Status:` lines. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` + `docs/adr/`, created lazily. See `docs/agents/domain.md`.

## After each ticket

1. Run the `ponytail:ponytail-review` skill on the diff and apply its cuts.
2. Run `make ci` (ruff check, ruff format check, pytest) and `make typecheck` (pyright). Done means
   zero findings, zero errors, all tests green. Pyright is local only: CI's sync skips the
   training/optimize groups its imports need.

Every change ships its unit tests and updates the matching `docs/*.md`.

## Code style

- Let names and structure explain the code. Keep docstrings and comments to one line, and only for
  a non-obvious why.
- Use f-strings in every `logger.*` / `print` call. Keep ruff's `G` rules unselected (G004 bans
  this).
- Type-hint public functions. Use pydantic models for config and structured I/O.
- Draw diagrams in docs as Mermaid.

## Dependencies

- Pin every package to an exact version and every image to a specific tag. `uv.lock` pins the
  rest; CI syncs `--locked`.
- Look up the current latest stable version before adding or bumping a package, Action or image.
- Declare every directly imported package in `pyproject.toml`, in the group that uses it.
- Lazy-import heavy deps (torch, ultralytics, tensorrt, …) inside functions, so every module
  imports in CI's CPU-only sync.
- Docker images opt into groups (`--no-default-groups --group <x>`). CI subtracts them
  (`--no-group training`), so a new group's tests run in CI by default.
- Beam caps numpy, pyarrow, pillow and cryptography lock-wide. Expect downgrades on `uv lock`, and
  rebuild the images after relocking.

## Configuration

- Entry points read `mlops_cv.config.get_settings()`. Never hard-code URIs or credentials.
- One `.env` (copied from `.env.example`) plus the OS env, which wins. Nested keys use `__`
  (`DATA__RAW_DIR`). See `docs/config.md`.
- A value that a compose env file or `prometheus.yml` repeats from Python keeps both copies, pinned
  by a golden test (`tests/unit/test_stack_constants.py`, `test_observability_config.py`). Add each
  new pair there.

## Owned contracts

Each module below owns a persistence or wire format, pinned by a golden test. Build those strings
and payloads through the module, never by hand at a call site.

- `tracking/metric_keys.py`: MLflow metric keys. The gate reads champions back by these keys, so a
  rename silently turns every gate into a bootstrap win.
- `tracking/client.py`: MLflow session setup; the only place a registry client is built.
- `tracking/champion.py`: champion alias → model version → training run.
- `tracking/records.py`: child record runs under a training run.
- `tracking/data_version.py`: the dataset profile published as a data version.
- `optimize/variants.py`: variant names and registry tag grammar (the Pi decodes these).
- `orchestration/handoff.py`: DAG ↔ container argv and last-stdout-line JSON.
- `tracking/edge_record.py`: the edge device's detections dump, package-version tags and the
  gate's measured tag on the record run.
- `streaming/messages.py`, `serving/schemas.py`: bus and HTTP schemas.
- `data/subset.py`: subset on-disk layout and CSV rows.
- `pipelines/profiling.py`: drift-baseline columns.

`metric_keys`, `variants`, `subset`, `profiling`, `provenance` and `handoff` stay stdlib +
pydantic + yaml: the Airflow and serving images import them.

## Invariants

- Entry-point `main`s wear `startup.exits_on_startup_error`. Raise `StartupError` naming the
  command that fixes it; the process exits 2.
- Writers call `tracking.client.connect(settings)` before importing mlflow. Read-only processes
  (service, consumers, monitor) go through `tracking.champion.resolve`.
- Services resolve the champion once at startup. A promotion takes effect on restart.
- A container task's last stdout line is its JSON result (the Airflow XCom). Print nothing after it.
- Airflow `@task`s are thin wrappers around `src/mlops_cv`. Pass small values via XCom and large
  artifacts via MLflow URIs.
- Metrics come only from ultralytics' metric code: `model.val()`, or
  `evaluation.score_detections` feeding stored detections through its validator. Never hand-roll
  AP math.
- Thresholds (anomaly, drift bars, `--shift` defaults) are measured values. Remeasure them when the
  champion, the demo clips or the inference path change. Trust `Settings` over numbers in prose.
- `serving/prepost.py` matches ultralytics' pre/post byte-for-byte against golden fixtures. Keep
  `cv2.resize`.

## Gotchas

- ultralytics resolves a relative dataset-YAML `path:` against its global `datasets_dir`, not the
  repo. Use the YAML the ingest pipeline generates (absolute `path:`).
- MLflow 3: register a raw `.pt` with `MlflowClient.create_model_version(source=…)`; fluent
  `register_model` errors on a bare artifact. Open `mlflow.start_run()` before `model.train()` so
  ultralytics' callback reuses the run.
- ultralytics' per-class `maps` back-fills absent classes with the overall mAP. Iterate
  `ap_class_index`.
- MLflow checks the `Host` header. A container calling `http://mlflow:5000` gets 403 unless
  `MLFLOW_SERVER_ALLOWED_HOSTS` lists it.
- onnxruntime finds CUDA through the system loader. On the host, `make serve` puts the venv's
  `nvidia/*/lib` on `LD_LIBRARY_PATH`; the images register them with `ldconfig`.
- Airflow DockerOperator mount sources are host paths. Mount them at the same absolute path inside
  the container, so the dataset YAML's absolute `path:` resolves.
- Start consumers before the producer: with `--offset-reset latest`, messages on never-read
  partitions are skipped.
- Compare GPU memory as deltas, never absolutes. The CUDA context floor dwarfs the gap between
  variants and ranks them backwards.
- The compose stacks join the MLflow stack's network; the `make *-up` targets start MLflow first.
