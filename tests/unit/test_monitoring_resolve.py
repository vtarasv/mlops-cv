"""Champion -> data version -> drift baseline: the monitor's fail-fast startup paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlops_cv.config import Settings
from mlops_cv.monitoring import resolve as monitoring_resolve
from mlops_cv.monitoring.resolve import resolve_baseline
from mlops_cv.pipelines import profiling
from mlops_cv.startup import StartupError
from mlops_cv.tracking.data_version import RUN_TAG


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    """A downloaded Profile directory, exactly as the data version published it."""
    local = tmp_path / "profile"
    local.mkdir()
    (local / profiling.PROFILE_JSON).write_text("{}", encoding="utf-8")
    (local / profiling.DRIFT_BASELINE_CSV).write_text(
        f"{profiling.baseline_header()}\nseqA,12,116.2,46.3,1748.6\nseqB,8,90.0,30.0,900.0\n",
        encoding="utf-8",
    )
    return local


@pytest.fixture
def champion(registry, monkeypatch: pytest.MonkeyPatch, profile_dir: Path):
    """A promoted champion whose training run names a live data version."""
    version = registry.promote(version="8", run_id="train-run")
    registry.runs["train-run"].data.tags[RUN_TAG] = "data-run"
    monkeypatch.setattr(monitoring_resolve, "download", lambda uri: profile_dir)
    return version


def test_resolves_the_champions_baseline_from_the_alias_alone(registry, champion) -> None:
    resolved = resolve_baseline(Settings(), registry=registry)

    assert [scene.sequence for scene in resolved.scenes] == ["seqA", "seqB"]
    assert resolved.scenes[0].means["brightness"] == pytest.approx(116.2)
    assert resolved.data_run_id == "data-run"
    assert resolved.training_run_id == "train-run"
    assert resolved.model.name == "aerial-object-detector"
    assert resolved.model.version == "8"


def test_the_walk_starts_at_the_champion_alias(registry, champion) -> None:
    """Restart-to-pick-up: the monitor judges whoever is champion now, like the service."""
    resolve_baseline(Settings(), registry=registry)
    assert registry.asked == [("alias", "aerial-object-detector", "champion")]


def test_the_baseline_is_read_from_the_data_versions_profile(
    registry, monkeypatch: pytest.MonkeyPatch, champion, profile_dir: Path
) -> None:
    """The pointer is followed to the run the training run names, not to a local directory."""
    asked: list[str] = []

    def download(uri: str) -> Path:
        asked.append(uri)
        return profile_dir

    monkeypatch.setattr(monitoring_resolve, "download", download)
    resolve_baseline(Settings(), registry=registry)
    assert asked == ["runs:/data-run/profile"]


def test_no_champion_fails_fast_with_the_training_hint(registry) -> None:
    with pytest.raises(StartupError) as exc:
        resolve_baseline(Settings(), registry=registry)
    assert "make train" in str(exc.value)


def test_a_training_run_naming_no_data_version_names_the_command_that_fixes_it(
    registry, champion
) -> None:
    """A model trained before data versions existed, or on an unprofiled Subset."""
    del registry.runs["train-run"].data.tags[RUN_TAG]
    with pytest.raises(StartupError) as exc:
        resolve_baseline(Settings(), registry=registry)
    message = str(exc.value)
    assert "train-run" in message  # which run
    assert RUN_TAG in message  # which tag it lacks
    assert "make profile" in message  # and how to produce one


def test_a_champion_naming_no_training_run_is_a_named_refusal(registry, champion) -> None:
    """A version registered outside a tracked run: the walk has no second step to take."""
    champion.run_id = None
    with pytest.raises(StartupError) as exc:
        resolve_baseline(Settings(), registry=registry)
    message = str(exc.value)
    assert "8" in message  # which model version
    assert "make train" in message


def test_a_data_version_that_is_gone_is_a_named_refusal(
    registry, monkeypatch: pytest.MonkeyPatch, champion
) -> None:
    """The price of a pointer: a pruned run must read as a refusal, not an empty baseline."""
    from mlflow.exceptions import MlflowException

    def missing(uri: str) -> Path:
        raise MlflowException(f"Run 'data-run' not found: {uri}")

    monkeypatch.setattr(monitoring_resolve, "download", missing)
    with pytest.raises(StartupError) as exc:
        resolve_baseline(Settings(), registry=registry)
    message = str(exc.value)
    assert "data-run" in message
    assert "make profile" in message


def test_a_profile_without_a_baseline_is_a_named_refusal(
    registry, monkeypatch: pytest.MonkeyPatch, champion, tmp_path: Path
) -> None:
    """A Profile published before the baseline existed: stale, and the fix is re-profiling."""
    empty = tmp_path / "old-profile"
    empty.mkdir()
    (empty / profiling.PROFILE_JSON).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(monitoring_resolve, "download", lambda uri: empty)
    with pytest.raises(StartupError) as exc:
        resolve_baseline(Settings(), registry=registry)
    message = str(exc.value)
    assert profiling.DRIFT_BASELINE_CSV in message
    assert "make profile" in message


def test_a_baseline_this_build_cannot_read_is_a_refusal_too(
    registry, monkeypatch: pytest.MonkeyPatch, champion, tmp_path: Path
) -> None:
    """A baseline written under different columns: the emit↔parse contract moved, so re-profile."""
    other_columns = tmp_path / "other-columns"
    other_columns.mkdir()
    (other_columns / profiling.DRIFT_BASELINE_CSV).write_text(
        "sequence,n_frames,brightness\nseqA,3,10.0\n", encoding="utf-8"
    )
    monkeypatch.setattr(monitoring_resolve, "download", lambda uri: other_columns)
    with pytest.raises(StartupError, match="make profile"):
        resolve_baseline(Settings(), registry=registry)


def test_an_empty_baseline_is_a_refusal_too(
    registry, monkeypatch: pytest.MonkeyPatch, champion, tmp_path: Path
) -> None:
    """A header-only CSV would otherwise derive thresholds from nothing."""
    headers_only = tmp_path / "headers-only"
    headers_only.mkdir()
    (headers_only / profiling.DRIFT_BASELINE_CSV).write_text(
        f"{profiling.baseline_header()}\n", encoding="utf-8"
    )
    monkeypatch.setattr(monitoring_resolve, "download", lambda uri: headers_only)
    with pytest.raises(StartupError, match="make profile"):
        resolve_baseline(Settings(), registry=registry)
