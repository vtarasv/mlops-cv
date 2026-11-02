"""Fetch a labeling trial's annotation files for the annotation editor; prints the loader's args.

They land in ``runs/autolabel/<run name>/annotations/``, which the FiftyOne App mounts as is.
``make autolabel-view RUN=<run id>`` then loads them there with ``docker/fiftyone/load_trial.py``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mlops_cv.data.subset import raw_split_dir
from mlops_cv.startup import StartupError, exits_on_startup_error
from mlops_cv.tuning.label_eval import ANNOTATIONS, AUTOLABEL_EXPERIMENT, RUNS_DIR, SPLIT_TAG


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download a labeling trial's annotation files.")
    p.add_argument("run_id", help=f"a run in MLflow experiment {AUTOLABEL_EXPERIMENT}")
    args = p.parse_args(argv)

    from mlops_cv.config import get_settings
    from mlops_cv.tracking import client

    settings = get_settings()
    mlflow = client.connect(settings, experiment=False)
    run = mlflow.get_run(args.run_id)
    split = run.data.tags.get(SPLIT_TAG)
    if split is None:
        raise StartupError(
            f"run {args.run_id} is no labeling trial: pass a run id from {AUTOLABEL_EXPERIMENT}"
        )
    out = RUNS_DIR.resolve() / run.info.run_name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    downloaded = mlflow.artifacts.download_artifacts(
        run_id=args.run_id, artifact_path=ANNOTATIONS, dst_path=str(out)
    )
    print(settings.data.raw_dir.resolve() / raw_split_dir(split), Path(downloaded).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
