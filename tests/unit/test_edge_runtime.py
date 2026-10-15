"""Pins the NCNN decode against ultralytics' NMS of a real ``out0``, without ncnn or torch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlops_cv.edge.runtime import decode

FIXTURE = Path(__file__).parent / "fixtures" / "ncnn_decode_golden.npz"


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(FIXTURE) as data:
        return dict(data)


@pytest.mark.parametrize(
    ("name", "conf", "multi_label"), [("predict", 0.25, False), ("val", 0.001, True)]
)
def test_decode_matches_ultralytics_nms(golden, name: str, conf: float, multi_label: bool) -> None:
    got = decode(golden["out0"], conf, 0.7, multi_label=multi_label)
    assert got.dtype == np.float32
    assert np.array_equal(got, golden[name])


def test_scores_below_conf_give_no_detections(golden) -> None:
    assert decode(golden["out0"], 1.0, 0.7).shape == (0, 6)


def test_max_det_caps_the_kept_boxes(golden) -> None:
    got = decode(golden["out0"], 0.001, 0.7, max_det=5, multi_label=True)
    assert np.array_equal(got, golden["val"][:5])


@pytest.mark.parametrize("shape", [(1, 7, 2100), (4, 2100), (2100,)])
def test_wrong_shape_raises(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="anchors"):
        decode(np.zeros(shape, np.float32), 0.25, 0.7)


# Two zero-area boxes on one point (IoU 0/0) and a real box elsewhere, all class 0.
ZERO_AREA_OUT0 = np.array(
    [[10, 10, 50], [10, 10, 50], [0, 0, 8], [0, 0, 8], [0.9, 0.8, 0.7], [0, 0, 0], [0, 0, 0]],
    np.float32,
)


@pytest.mark.filterwarnings("error")
def test_nan_iou_between_zero_area_boxes_keeps_both() -> None:
    got = decode(ZERO_AREA_OUT0, 0.25, 0.7)
    assert got[:, 4].tolist() == pytest.approx([0.9, 0.8, 0.7])


def test_nan_iou_matches_ultralytics_nms() -> None:
    torch = pytest.importorskip("torch")
    nms = pytest.importorskip("ultralytics.utils.nms")
    (golden,) = nms.non_max_suppression(torch.from_numpy(ZERO_AREA_OUT0[None].copy()), 0.25, 0.7)
    assert np.array_equal(decode(ZERO_AREA_OUT0, 0.25, 0.7), golden.numpy())
