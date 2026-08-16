"""The measured separation between normal and shifted footage, pinned."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlops_cv.monitoring.drift import DriftReference, EpisodeRule
from mlops_cv.pipelines.profiling import BaselineScene

FIXTURE = Path(__file__).parent / "fixtures" / "drift_golden.json"

# Derived from the 56 training scenes at startup — never configured. Blur's bar is higher than
# the others' because it is scored in log space, where the scene cloud is wider.
THRESHOLDS = {"brightness": 2.60, "contrast": 2.35, "blur": 3.20}

# (clip, mode) -> (brightness, contrast, blur) scores, and the statistics that crossed.
GOLDEN: dict[tuple[str, str], tuple[tuple[float, float, float], tuple[str, ...]]] = {
    ("uav0000073_00600_v", "normal"): ((0.68, 0.68, 0.56), ()),
    ("uav0000161_00000_v", "normal"): ((1.23, 0.31, 1.32), ()),
    ("uav0000355_00001_v", "normal"): ((0.99, 1.18, 1.12), ()),
    # Defocus: malignant (it destroys the champion's detections) and unmistakable in log space.
    ("uav0000073_00600_v", "defocus-r4"): ((0.68, 0.28, 5.86), ("blur",)),
    ("uav0000161_00000_v", "defocus-r4"): ((1.23, 0.50, 4.34), ("blur",)),
    ("uav0000355_00001_v", "defocus-r4"): ((0.99, 1.99, 4.07), ("blur",)),
    # Photometric: benign (the model absorbs it) but still an input-side alarm — which is what
    # makes the alarm's meaning honest: the camera changed, not the model failed.
    ("uav0000161_00000_v", "dark-x0.55"): ((2.76, 2.02, 0.10), ("brightness",)),
    ("uav0000355_00001_v", "dark-x0.55"): ((1.55, 2.83, 0.19), ("contrast",)),
    ("uav0000161_00000_v", "bright-x1.6"): ((0.61, 2.40, 1.90), ("contrast",)),
    ("uav0000355_00001_v", "bright-x1.6"): ((3.94, 0.40, 1.47), ("brightness",)),
    # A clip sitting near the centre of the cloud needs a larger shift than ±45-60% to fire.
    ("uav0000073_00600_v", "dark-x0.55"): ((2.46, 1.82, 2.07), ()),
    ("uav0000073_00600_v", "bright-x1.6"): ((1.31, 2.24, 0.05), ()),
}

# The same defocus scored symmetrically — the rule the log space earns its place against.
SYMMETRIC_DEFOCUS = {
    "uav0000073_00600_v": 1.93,
    "uav0000161_00000_v": 1.87,
    "uav0000355_00001_v": 1.85,
}
SYMMETRIC_BLUR_THRESHOLD = 2.36


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scenes(golden: dict) -> list[BaselineScene]:
    return [BaselineScene(**scene) for scene in golden["baseline"]]


@pytest.fixture(scope="module")
def windows(golden: dict) -> dict[tuple[str, str], dict[str, float]]:
    return {(w["clip"], w["mode"]): w["means"] for w in golden["windows"]}


def test_the_thresholds_are_derived_from_the_training_scenes(scenes: list[BaselineScene]) -> None:
    reference = DriftReference.from_scenes(scenes)

    assert reference.n_scenes == 56
    for metric, expected in THRESHOLDS.items():
        assert reference.thresholds[metric] == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(("clip", "mode"), list(GOLDEN))
def test_the_measured_scores_and_verdicts_hold(
    scenes: list[BaselineScene],
    windows: dict[tuple[str, str], dict[str, float]],
    clip: str,
    mode: str,
) -> None:
    expected_scores, expected_crossed = GOLDEN[(clip, mode)]

    verdict = DriftReference.from_scenes(scenes).judge(windows[(clip, mode)])

    assert verdict.crossed == expected_crossed
    assert (
        verdict.scores["brightness"],
        verdict.scores["contrast"],
        verdict.scores["blur"],
    ) == pytest.approx(expected_scores, abs=0.01)


def test_normal_footage_never_reaches_its_bars(
    scenes: list[BaselineScene], windows: dict[tuple[str, str], dict[str, float]]
) -> None:
    """No false alarms, with headroom: ordinary traffic peaks well under the widest scene."""
    reference = DriftReference.from_scenes(scenes)

    peak = 0.0
    for (clip, mode), means in windows.items():
        if mode != "normal":
            continue
        verdict = reference.judge(means)
        assert not verdict.drifted, clip
        peak = max(peak, max(verdict.scores.values()))

    assert peak == pytest.approx(1.32, abs=0.01)  # against bars of 2.35-3.20


def test_scoring_the_defocus_symmetrically_could_not_fire(
    scenes: list[BaselineScene], windows: dict[tuple[str, str], dict[str, float]]
) -> None:
    """Why blur is scored in log space, on the shift that actually breaks the model."""
    symmetric = DriftReference.from_scenes(scenes, log_scale=())
    assert symmetric.thresholds["blur"] == pytest.approx(SYMMETRIC_BLUR_THRESHOLD, abs=0.01)

    for clip, expected in SYMMETRIC_DEFOCUS.items():
        verdict = symmetric.judge(windows[(clip, "defocus-r4")])
        assert verdict.scores["blur"] == pytest.approx(expected, abs=0.01)
        assert not verdict.drifted, clip

    # The cap is structural, not a near miss: no blur reading at all can clear the bar downward.
    floor = symmetric.judge({"brightness": 116.2, "contrast": 46.3, "blur": 0.0})
    assert floor.scores["blur"] < SYMMETRIC_BLUR_THRESHOLD


def test_a_sustained_defocus_opens_exactly_one_episode(
    scenes: list[BaselineScene], windows: dict[tuple[str, str], dict[str, float]]
) -> None:
    """End to end over measured windows: normal traffic, then the shift, then recovery."""
    reference = DriftReference.from_scenes(scenes)
    rule = EpisodeRule()
    clip = "uav0000161_00000_v"
    stream = ["normal", "normal", "defocus-r4", "defocus-r4", "defocus-r4", "normal"]

    episodes = [rule.observe(reference.judge(windows[(clip, mode)])) for mode in stream]

    assert [e is not None for e in episodes] == [False, False, False, True, False, False]
    opened = episodes[3]
    assert opened is not None
    assert opened.crossed == ("blur",)
    assert opened.scores["blur"] > opened.thresholds["blur"]
