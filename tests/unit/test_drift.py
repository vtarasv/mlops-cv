"""The drift statistic, its thresholds and the episode rule — pure logic, no broker, no registry."""

from __future__ import annotations

import math

import pytest

from mlops_cv.monitoring.drift import DriftReference, EpisodeRule, window_means
from mlops_cv.pipelines.profiling import BaselineScene

# Clouds whose centre and spread are exact: brightness/contrast 100/50 ± 10 (population sd),
# blur log-spaced around 1000 so its log-space centre is exact too.
BRIGHTNESS = [90.0, 100.0, 110.0]
CONTRAST = [40.0, 50.0, 60.0]
BLUR = [100.0, 1000.0, 10000.0]


def _scenes(
    brightness: list[float] | None = None,
    contrast: list[float] | None = None,
    blur: list[float] | None = None,
) -> list[BaselineScene]:
    columns = (brightness or BRIGHTNESS, contrast or CONTRAST, blur or BLUR)
    return [
        BaselineScene(
            sequence=f"seq{i}",
            n_frames=10,
            means={"brightness": b, "contrast": c, "blur": u},
        )
        for i, (b, c, u) in enumerate(zip(*columns, strict=True))
    ]


def _means(brightness: float = 100.0, contrast: float = 50.0, blur: float = 1000.0) -> dict:
    return {"brightness": brightness, "contrast": contrast, "blur": blur}


def test_a_windows_score_is_its_distance_from_the_scene_cloud_in_sds() -> None:
    """The statistic: how far this window's mean sits from the cloud of training-scene means."""
    reference = DriftReference.from_scenes(_scenes())

    verdict = reference.judge(_means(brightness=125.0, contrast=50.0))

    assert verdict.scores["brightness"] == pytest.approx(25.0 / math.sqrt(200 / 3))
    assert verdict.scores["contrast"] == pytest.approx(0.0)  # dead centre


def test_the_score_is_symmetric_about_the_cloud_centre() -> None:
    """A collapse and a rise of the same size are equally far from normal."""
    reference = DriftReference.from_scenes(_scenes())

    below = reference.judge(_means(brightness=80.0)).scores["brightness"]
    above = reference.judge(_means(brightness=120.0)).scores["brightness"]

    assert below == pytest.approx(above)


def test_ratio_scale_statistics_are_scored_in_log_space() -> None:
    """``blur`` is variance-like with a hard floor at zero, so distance is multiplicative."""
    reference = DriftReference.from_scenes(_scenes())

    # The cloud is 1000 x/÷ 10: dividing and multiplying by ten are the same distance from it,
    # which arithmetically (-900 vs +9000) they are not.
    assert reference.judge(_means(blur=1000.0)).scores["blur"] == pytest.approx(0.0)
    assert reference.judge(_means(blur=100.0)).scores["blur"] == pytest.approx(math.sqrt(1.5))
    assert reference.judge(_means(blur=10000.0)).scores["blur"] == pytest.approx(math.sqrt(1.5))


def test_which_statistics_are_ratio_scale_is_an_argument_not_a_hidden_branch() -> None:
    """Extendable by naming a statistic, and overridable — which is what pins the log rule."""
    symmetric = DriftReference.from_scenes(_scenes(), log_scale=())

    # Same cloud, scored symmetrically: ÷10 and ×10 stop being the same distance.
    below = symmetric.judge(_means(blur=100.0)).scores["blur"]
    above = symmetric.judge(_means(blur=10000.0)).scores["blur"]
    assert below != pytest.approx(above)


def test_a_collapse_toward_zero_stays_detectable_in_log_space() -> None:
    """The defect the log rule fixes: a symmetric score on a floored quantity caps out."""
    scenes = _scenes()
    log_scored = DriftReference.from_scenes(scenes)
    symmetric = DriftReference.from_scenes(scenes, log_scale=())

    collapsed = _means(blur=1.0)  # a near-total collapse of edge energy

    assert log_scored.judge(collapsed).crossed == ("blur",)
    # Symmetrically, the whole range below the centre is worth less than centre/sd — the score
    # cannot reach its own bar however far the statistic falls.
    floor_score = symmetric.judge(_means(blur=0.0)).scores["blur"]
    assert symmetric.judge(collapsed).scores["blur"] <= floor_score
    assert not symmetric.judge(collapsed).crossed


def test_thresholds_are_the_widest_distance_the_training_scenes_themselves_reach() -> None:
    reference = DriftReference.from_scenes(_scenes())

    # The extreme scenes of a 3-point cloud sit at sqrt(3/2) sd; blur's do too, in log space.
    assert reference.thresholds["brightness"] == pytest.approx(math.sqrt(1.5))
    assert reference.thresholds["contrast"] == pytest.approx(math.sqrt(1.5))
    assert reference.thresholds["blur"] == pytest.approx(math.sqrt(1.5))


def test_the_margin_multiplies_the_derived_thresholds_and_defaults_to_one() -> None:
    plain = DriftReference.from_scenes(_scenes())
    widened = DriftReference.from_scenes(_scenes(), margin=1.5)

    assert widened.thresholds["brightness"] == pytest.approx(plain.thresholds["brightness"] * 1.5)


def test_the_most_extreme_training_scene_is_not_itself_drifted() -> None:
    """The bar is the widest normal scene, so that scene must read as normal."""
    reference = DriftReference.from_scenes(_scenes())

    verdict = reference.judge(_means(brightness=110.0, contrast=60.0, blur=10000.0))

    assert verdict.scores["brightness"] == pytest.approx(verdict.thresholds["brightness"])
    assert not verdict.drifted


def test_a_window_is_drifted_when_any_single_statistic_crosses() -> None:
    """Any-one, not a majority: a night shift moves brightness hard and blur not at all."""
    reference = DriftReference.from_scenes(_scenes())

    verdict = reference.judge(_means(brightness=200.0))

    assert verdict.crossed == ("brightness",)
    assert verdict.drifted
    assert verdict.scores["contrast"] < verdict.thresholds["contrast"]


def test_a_verdict_carries_every_score_beside_its_own_bar() -> None:
    """What the panel and the evidence run both need: how close, not just whether."""
    reference = DriftReference.from_scenes(_scenes())

    verdict = reference.judge(_means(brightness=200.0, contrast=200.0))

    assert sorted(verdict.scores) == ["blur", "brightness", "contrast"]
    assert sorted(verdict.thresholds) == ["blur", "brightness", "contrast"]
    assert verdict.crossed == ("brightness", "contrast")  # baseline column order


def test_a_window_missing_a_statistic_is_a_named_error() -> None:
    reference = DriftReference.from_scenes(_scenes())

    with pytest.raises(ValueError, match="blur"):
        reference.judge({"brightness": 100.0, "contrast": 50.0})


def test_a_baseline_with_no_spread_cannot_derive_a_threshold() -> None:
    """One scene, or 56 identical ones: every distance is 0/0 and every bar meaningless."""
    with pytest.raises(ValueError, match="brightness"):
        DriftReference.from_scenes(_scenes(brightness=[100.0, 100.0, 100.0]))


def test_an_empty_baseline_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        DriftReference.from_scenes([])


def test_naming_a_statistic_that_is_not_scored_ratio_scale_is_refused() -> None:
    """A misspelling would silently revert a statistic to symmetric scoring — invisibly."""
    with pytest.raises(ValueError, match="blurr"):
        DriftReference.from_scenes(_scenes(), log_scale=("blurr",))


def test_a_margin_that_cannot_widen_a_threshold_is_refused() -> None:
    """A zero margin puts every bar at zero, so every window on earth reads as drifted."""
    with pytest.raises(ValueError, match="positive"):
        DriftReference.from_scenes(_scenes(), margin=0.0)


def test_window_means_average_the_drift_statistics_and_ignore_the_rest() -> None:
    """Per-frame readings come from the profiling function, which computes more than three."""
    readings = [
        {"brightness": 90.0, "contrast": 40.0, "blur": 900.0, "width": 1920, "dhash": "ff"},
        {"brightness": 110.0, "contrast": 60.0, "blur": 1100.0, "width": 1920, "dhash": "ee"},
    ]

    assert window_means(readings) == {"brightness": 100.0, "contrast": 50.0, "blur": 1000.0}


def test_an_empty_window_has_no_mean() -> None:
    with pytest.raises(ValueError, match="empty"):
        window_means([])


# --- the episode rule ------------------------------------------------------------------------


def _verdict(reference: DriftReference, *, drifted: bool):  # noqa: ANN202 - test helper
    return reference.judge(_means(brightness=200.0 if drifted else 100.0))


def test_one_drifted_window_is_not_news() -> None:
    reference = DriftReference.from_scenes(_scenes())
    rule = EpisodeRule()

    assert rule.observe(_verdict(reference, drifted=True)) is None


def test_an_episode_opens_on_the_second_consecutive_drifted_window() -> None:
    reference = DriftReference.from_scenes(_scenes())
    rule = EpisodeRule()

    rule.observe(_verdict(reference, drifted=True))
    episode = rule.observe(_verdict(reference, drifted=True))

    assert episode is not None
    assert episode.crossed == ("brightness",)
    assert episode.scores["brightness"] > episode.thresholds["brightness"]


def test_a_sustained_shift_is_reported_once() -> None:
    reference = DriftReference.from_scenes(_scenes())
    rule = EpisodeRule()

    rule.observe(_verdict(reference, drifted=True))
    assert rule.observe(_verdict(reference, drifted=True)) is not None
    assert rule.observe(_verdict(reference, drifted=True)) is None
    assert rule.observe(_verdict(reference, drifted=True)) is None


def test_the_rule_re_arms_only_after_a_clean_window() -> None:
    reference = DriftReference.from_scenes(_scenes())
    rule = EpisodeRule()

    rule.observe(_verdict(reference, drifted=True))
    assert rule.observe(_verdict(reference, drifted=True)) is not None
    assert rule.observe(_verdict(reference, drifted=False)) is None  # clears silently
    # A new episode needs its own two consecutive windows.
    assert rule.observe(_verdict(reference, drifted=True)) is None
    assert rule.observe(_verdict(reference, drifted=True)) is not None


def test_a_clean_window_breaks_the_run_before_an_episode_opens() -> None:
    """Alternating windows are noise, not a shift — no cooldown needed to say so."""
    reference = DriftReference.from_scenes(_scenes())
    rule = EpisodeRule()

    for drifted in (True, False, True, False, True):
        assert rule.observe(_verdict(reference, drifted=drifted)) is None


def test_how_many_consecutive_windows_open_an_episode_is_a_setting() -> None:
    reference = DriftReference.from_scenes(_scenes())
    rule = EpisodeRule(consecutive=3)

    assert rule.observe(_verdict(reference, drifted=True)) is None
    assert rule.observe(_verdict(reference, drifted=True)) is None
    assert rule.observe(_verdict(reference, drifted=True)) is not None
