# macOS dev box

A Mac with Apple Silicon and Docker Desktop is a dev box for code, tests, data pipelines and the
CPU stacks. Training, the CT DAG, optimization and serving stay on the Linux/NVIDIA host.

## What doesn't work, and why

- **The CT DAG's train, evaluate and optimize tasks, and the serving stack** request NVIDIA
  devices. Docker Desktop's Linux VM has no Metal driver, so no container reaches the Mac GPU.
- **`onnxruntime-gpu` and `tensorrt-cu13`** are Linux-only (`sys_platform == 'linux'` markers),
  so `make optimize` and `make serve` can't run.
- **`gpu`-marked tests** need CUDA.
- **`make typecheck`** reports two missing imports (`tensorrt`, `onnxruntime`).

## Port 5000

macOS AirPlay Receiver listens on `*:5000`, so `make mlflow-up` fails with "address already in
use". Put `export MLFLOW_PORT=5050` in your shell profile and set
`MLFLOW__TRACKING_URI=http://localhost:5050` in `.env`. The shell value overrides
`.env.mlflow`, and it has to be there for every `make` call: each stack target re-runs
`mlflow-up`.

## Relocking

`tensorrt-cu13-libs` ships only a wheel-stub sdist, which can't build off Linux. The
`[[tool.uv.dependency-metadata]]` entry in `pyproject.toml` hands uv its metadata, so a Mac locks
without building it. Its version follows the `tensorrt-cu13` pin, and `make ci` fails if the two drift.

`[tool.uv] required-environments` makes any relock fail if either the Linux host or the Mac stops resolving.
