"""Fetch a labeling trial's auto-labels for the annotation editor; prints the file's path.

The file lands in its scoring set's ``trials/``, which the FiftyOne App mounts at the same path.
``make autolabel-view RUN=<run id>`` then loads it there with ``docker-compose/fiftyone``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mlops_cv.autolabel import visdrone_vid
from mlops_cv.autolabel.run import AUTOLABEL_EXPERIMENT, SET_TAG, autolabels_file
from mlops_cv.startup import StartupError, exits_on_startup_error


def trial_path(autolabel_dir: Path, scoring_set: str, run_name: str) -> Path:
    return visdrone_vid.scoring_dir(autolabel_dir, scoring_set) / "trials" / f"{run_name}.json"


@exits_on_startup_error
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download a labeling trial's auto-labels.")
    p.add_argument("run_id", help=f"a run in MLflow experiment {AUTOLABEL_EXPERIMENT}")
    args = p.parse_args(argv)

    from mlops_cv.config import get_settings
    from mlops_cv.tracking import client

    settings = get_settings()
    mlflow = client.connect(settings, experiment=False)
    run = mlflow.get_run(args.run_id)
    scoring_set = run.data.tags.get(SET_TAG)
    if scoring_set is None:
        raise StartupError(
            f"run {args.run_id} is no labeling trial: pass a run id from {AUTOLABEL_EXPERIMENT}"
        )
    out = trial_path(settings.data.autolabel_dir, scoring_set, run.info.run_name)
    out.parent.mkdir(exist_ok=True)
    downloaded = mlflow.artifacts.download_artifacts(
        run_id=args.run_id, artifact_path=autolabels_file(scoring_set), dst_path=str(out.parent)
    )
    Path(downloaded).replace(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
