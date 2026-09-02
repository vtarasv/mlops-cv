"""Unit tests for the run-metric key vocabulary (pure; no ultralytics, no mlflow, no numpy)."""

from __future__ import annotations

from types import SimpleNamespace

from mlops_cv.tracking.metric_keys import (
    DATA_PREFIX,
    DRIFT_EPISODE_METRIC,
    DRIFT_PREFIX,
    HEADLINE_ATTRS,
    OPTIMIZE_PREFIX,
    PRIMARY,
    TRAIN_PREFIX,
    data_metrics,
    drift_statistic_prefix,
    episode_metrics,
    headline_metrics,
    metric_key,
    per_class_metrics,
    variant_device_latency_prefix,
    variant_latency_prefix,
    variant_prefix,
)

NAMES = {0: "person", 1: "vehicle", 2: "two-three-wheeler"}


def _box(mp: float, mr: float, map50: float, map_: float) -> SimpleNamespace:
    """An ultralytics-``results.box`` stand-in exposing the four headline attributes."""
    return SimpleNamespace(mp=mp, mr=mr, map50=map50, map=map_)


def test_golden_spellings() -> None:
    # The persisted contract: change any of these and previously recorded runs stop matching.
    assert PRIMARY == "mAP50-95"
    assert TRAIN_PREFIX == "metrics"
    assert metric_key("test", PRIMARY) == "test/mAP50-95"
    assert list(HEADLINE_ATTRS) == ["precision", "recall", "mAP50", "mAP50-95"]


def test_headline_metrics_keys_and_values() -> None:
    out = headline_metrics(_box(0.8, 0.7, 0.6, 0.5))
    assert out == {
        "test/precision": 0.8,
        "test/recall": 0.7,
        "test/mAP50": 0.6,
        "test/mAP50-95": 0.5,
    }


def test_headline_metrics_prefix_override() -> None:
    out = headline_metrics(_box(1, 1, 1, 1), prefix="val")
    assert set(out) == {"val/precision", "val/recall", "val/mAP50", "val/mAP50-95"}


def test_headline_metrics_coerces_float() -> None:
    out = headline_metrics(_box(1, 0, 0, 0))  # ints in
    assert out["test/precision"] == 1.0
    assert isinstance(out["test/precision"], float)


def test_per_class_only_present_classes_emitted() -> None:
    # maps is class-id-indexed; ultralytics back-fills absent classes with the overall mAP, so the
    # value at index 1 (vehicle, not in ap_class_index) must be ignored, not logged.
    maps = [0.5, 0.99, 0.3]  # index 1 == 0.99 is the back-filled overall mAP
    out = per_class_metrics(maps, [0, 2], NAMES)
    assert out == {"metrics/mAP50-95/person": 0.5, "metrics/mAP50-95/two-three-wheeler": 0.3}


def test_per_class_prefix_override() -> None:
    out = per_class_metrics([0.4, 0.0, 0.0], [0], NAMES, prefix="test/mAP50-95")
    assert out == {"test/mAP50-95/person": 0.4}


def test_per_class_values_coerced_to_float() -> None:
    out = per_class_metrics([1, 0, 0], [0], NAMES)  # int in -> float out
    assert out == {"metrics/mAP50-95/person": 1.0}
    assert isinstance(out["metrics/mAP50-95/person"], float)


def test_variant_namespace_golden_spellings() -> None:
    assert OPTIMIZE_PREFIX == "optimize"
    assert variant_prefix("trt-fp16-640") == "optimize/trt-fp16-640"
    assert variant_latency_prefix("trt-fp16-640") == "optimize/trt-fp16-640/latency"
    assert (
        variant_device_latency_prefix("ncnn-fp16-320", "pi5")
        == "optimize/ncnn-fp16-320/latency/pi5"
    )


def test_device_latency_keys_never_collide_with_the_desktops() -> None:
    plain = metric_key(variant_latency_prefix("ncnn-fp16-320"), "p50_ms")
    device = metric_key(variant_device_latency_prefix("ncnn-fp16-320", "pi5"), "p50_ms")
    assert plain != device
    assert device.startswith(variant_latency_prefix("ncnn-fp16-320") + "/")


def test_variant_keys_can_never_collide_with_headline_keys() -> None:
    headline = {
        metric_key(split, name) for split in ("train", "val", "test") for name in HEADLINE_ATTRS
    }
    variant_keys = {
        metric_key(variant_prefix(variant), name)
        for variant in ("torch-fp32-640", "onnx-ort-fp32-640", "trt-fp16-640", "trt-int8-320")
        for name in HEADLINE_ATTRS
    }
    assert not (headline & variant_keys)


def test_optimize_prefix_is_not_a_split_name() -> None:
    """If a split were ever named 'optimize', the namespaces would overlap."""
    assert OPTIMIZE_PREFIX not in {"train", "val", "test"}


def test_data_version_count_golden_spellings() -> None:
    """A data version is read back by these keys, exactly as a model version is by its own."""
    assert DATA_PREFIX == "data"
    assert data_metrics({"n_frames": 1238, "n_baseline_sequences": 56}) == {
        "data/n_frames": 1238.0,
        "data/n_baseline_sequences": 56.0,
    }


def test_data_counts_are_floats_and_never_collide_with_detection_keys() -> None:
    out = data_metrics({"n_boxes": 12})
    assert isinstance(out["data/n_boxes"], float)
    headline = {
        metric_key(split, name) for split in ("train", "val", "test") for name in HEADLINE_ATTRS
    }
    assert not (headline & set(out))
    assert DATA_PREFIX not in {"train", "val", "test", OPTIMIZE_PREFIX}


def _verdict(scores: dict[str, float], thresholds: dict[str, float], crossed: tuple[str, ...]):  # noqa: ANN202
    """A drift-verdict stand-in: the three attributes the episode row is built from."""
    return SimpleNamespace(scores=scores, thresholds=thresholds, crossed=crossed)


def test_drift_episode_golden_spellings() -> None:
    """An episode series is read back by these keys across monitor restarts."""
    assert DRIFT_PREFIX == "drift"
    assert drift_statistic_prefix("brightness") == "drift/brightness"
    assert DRIFT_EPISODE_METRIC == "drift/n_crossed"


def test_episode_metrics_pairs_every_score_with_its_own_bar() -> None:
    verdict = _verdict(
        {"brightness": 3.5, "blur": 1.0}, {"brightness": 2.6, "blur": 3.2}, ("brightness",)
    )
    out = episode_metrics(verdict)
    assert out == {
        "drift/brightness/score": 3.5,
        "drift/brightness/threshold": 2.6,
        "drift/blur/score": 1.0,
        "drift/blur/threshold": 3.2,
        "drift/n_crossed": 1.0,
    }


def test_episode_metrics_always_writes_the_series_marker() -> None:
    """Its history is how a restarted monitor finds the next step, so it can never be absent."""
    out = episode_metrics(_verdict({}, {}, ()))
    assert out == {DRIFT_EPISODE_METRIC: 0.0}
    assert isinstance(out[DRIFT_EPISODE_METRIC], float)


def test_drift_keys_can_never_collide_with_the_keys_the_gate_reads() -> None:
    """The evidence lands on a child run; the gate reads the parent. Namespaces stay disjoint."""
    gate_keys = {
        metric_key(split, name) for split in ("train", "val", "test") for name in HEADLINE_ATTRS
    } | {metric_key(TRAIN_PREFIX, PRIMARY)}
    drift_keys = set(
        episode_metrics(_verdict({s: 1.0 for s in ("brightness", "contrast", "blur")}, {}, ()))
    )
    assert not (gate_keys & drift_keys)
    assert DRIFT_PREFIX not in {"train", "val", "test", OPTIMIZE_PREFIX, DATA_PREFIX, TRAIN_PREFIX}


def test_variant_headline_metrics_are_namespaced() -> None:
    box = _box(0.8, 0.7, 0.6, 0.4)
    out = headline_metrics(box, prefix=variant_prefix("trt-int8-320"))
    assert out["optimize/trt-int8-320/mAP50-95"] == 0.4
    assert not any(k.startswith(("test/", "val/", "train/")) for k in out)
