"""Golden pins + property tests for the serving pre/post-processing contract.

The npz fixture was generated once locally by ``fixtures/generate_prepost_golden.py``
with real ultralytics preprocessing; here the module is pinned against it byte for byte
without installing ultralytics — the fixture is the contract.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mlops_cv.serving import prepost

FIXTURE = Path(__file__).parent / "fixtures" / "prepost_golden.npz"
LETTERBOX_CASES = ["down", "up", "half", "narrow", "visdrone"]


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(FIXTURE) as data:
        return dict(data)


@pytest.mark.parametrize("name", LETTERBOX_CASES)
def test_letterbox_matches_ultralytics_byte_for_byte(golden, name: str) -> None:
    got = prepost.letterbox(golden[f"{name}_src"], int(golden[f"{name}_imgsz"]))
    assert got.dtype == np.uint8
    assert np.array_equal(got, golden[f"{name}_letterboxed"])


def test_input_tensor_matches_ultralytics_normalization(golden) -> None:
    got = prepost.to_input_tensor(golden["down_letterboxed"])
    assert got.dtype == np.float32
    assert got.flags["C_CONTIGUOUS"]
    assert np.array_equal(got, golden["down_tensor"])


def test_preprocess_composes_letterbox_and_normalization(golden) -> None:
    got = prepost.preprocess(golden["down_src"], int(golden["down_imgsz"]))
    assert got.shape == (1, 3, 64, 64)
    assert np.array_equal(got, golden["down_tensor"])


# "scale_boxes_rounding" pins ultralytics' inner round(src * gain): its source shape is
# chosen so omitting the round shifts the derived padding by one pixel.
@pytest.mark.parametrize("case", ["scale_boxes", "scale_boxes_rounding"])
def test_scale_boxes_matches_ultralytics(golden, case: str) -> None:
    boxes_in = golden["scale_boxes_in"]
    before = boxes_in.copy()
    got = prepost.scale_boxes(
        boxes_in,
        tuple(golden["scale_boxes_letterbox_shape"]),
        tuple(golden[f"{case}_source_shape"]),
    )
    assert np.array_equal(got, golden[f"{case}_out"])
    assert np.array_equal(boxes_in, before)  # never mutates its input


@pytest.mark.parametrize(
    "source_shape",
    [(756, 1344), (540, 960), (97, 53), (300, 300), (28, 40)],  # includes smaller-than-target
)
@pytest.mark.parametrize("imgsz", [640, 320])
def test_scale_boxes_round_trips_source_coordinates(source_shape, imgsz: int) -> None:
    src_h, src_w = source_shape
    rng = np.random.default_rng(hash((source_shape, imgsz)) % 2**32)
    x1 = rng.uniform(0, src_w * 0.9, size=32)
    y1 = rng.uniform(0, src_h * 0.9, size=32)
    boxes = np.stack(
        [x1, y1, x1 + rng.uniform(0.5, src_w * 0.1, 32), y1 + rng.uniform(0.5, src_h * 0.1, 32)],
        axis=1,
    ).astype(np.float32)

    # forward: source -> letterbox space, exactly as the letterboxed image is laid out
    gain = min(imgsz / src_h, imgsz / src_w)
    pad_x = round((imgsz - round(src_w * gain)) / 2 - 0.1)
    pad_y = round((imgsz - round(src_h * gain)) / 2 - 0.1)
    in_letterbox = boxes * gain
    in_letterbox[:, [0, 2]] += pad_x
    in_letterbox[:, [1, 3]] += pad_y

    back = prepost.scale_boxes(in_letterbox, (imgsz, imgsz), source_shape)
    assert np.allclose(back, boxes, atol=1e-2)


def test_scale_boxes_clips_to_source_bounds() -> None:
    boxes = np.array([[-500.0, -500.0, 10_000.0, 10_000.0]], dtype=np.float32)
    out = prepost.scale_boxes(boxes, (640, 640), (756, 1344))
    assert np.array_equal(out, [[0.0, 0.0, 1344.0, 756.0]])


def test_confidence_filter_keeps_the_boundary() -> None:
    output = np.zeros((1, 4, 6), dtype=np.float32)
    output[0, :, :4] = [100, 100, 200, 200]
    output[0, :, 4] = [0.51, 0.5, 0.49999997, 0.0]
    output[0, :, 5] = [0, 1, 2, 3]
    kept = prepost.postprocess(output, (640, 640), imgsz=640, conf_threshold=0.5)
    # >= threshold kept: 0.5 itself survives, the value one ulp below does not
    assert kept[:, 5].tolist() == [0.0, 1.0]
    assert kept[:, 4].tolist() == [np.float32(0.51), np.float32(0.5)]


def test_postprocess_rescales_into_source_pixels() -> None:
    # 1280x720 source -> gain 0.5, pads (0, 140): letterbox (320, 250, 480, 400)
    output = np.array([[[320.0, 250.0, 480.0, 400.0, 0.9, 2.0]]], dtype=np.float32)
    detections = prepost.postprocess(output, (720, 1280), imgsz=640, conf_threshold=0.25)
    assert detections.shape == (1, 6)
    assert np.allclose(detections[0], [640.0, 220.0, 960.0, 520.0, 0.9, 2.0])


def test_postprocess_accepts_squeezed_output_and_rejects_bad_shapes() -> None:
    squeezed = np.array([[0.0, 0.0, 10.0, 10.0, 0.9, 1.0]], dtype=np.float32)
    assert prepost.postprocess(squeezed, (640, 640)).shape == (1, 6)
    with pytest.raises(ValueError, match="batch of 1"):
        prepost.postprocess(np.zeros((2, 5, 6), dtype=np.float32), (640, 640))
    with pytest.raises(ValueError, match="max_det, 6"):
        prepost.postprocess(np.zeros((5, 4), dtype=np.float32), (640, 640))


def test_letterbox_rejects_non_rgb_input() -> None:
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        prepost.letterbox(np.zeros((64, 64), dtype=np.uint8))
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        prepost.letterbox(np.zeros((64, 64, 3), dtype=np.float32))


def test_letterbox_pads_with_gray() -> None:
    out = prepost.letterbox(np.zeros((100, 30, 3), dtype=np.uint8), 96)
    assert out.shape == (96, 96, 3)
    assert np.all(out[:, :30] == prepost.PAD_COLOR)  # left pad band
    assert np.all(out[:, -30:] == prepost.PAD_COLOR)  # right pad band


def test_load_image_decodes_to_rgb_array(tmp_path: Path) -> None:
    from PIL import Image  # test-side encoder only; the module decodes with cv2

    rgb = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    assert np.array_equal(prepost.load_image(buf.getvalue()), rgb)
    path = tmp_path / "img.png"
    path.write_bytes(buf.getvalue())
    assert np.array_equal(prepost.load_image(path), rgb)


def test_load_image_honors_exif_orientation() -> None:
    """cv2 decode applies EXIF rotation — the behavior eval's cv2.imread inputs had."""
    from PIL import Image

    img = Image.new("RGB", (40, 20), (10, 200, 30))
    exif = Image.Exif()
    exif[274] = 6  # Orientation: rotate 90 CW to display
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    assert prepost.load_image(buf.getvalue()).shape == (40, 20, 3)


def test_load_image_rejects_undecodable_bytes() -> None:
    with pytest.raises(ValueError, match="decode"):
        prepost.load_image(b"definitely not an image")


def test_module_imports_without_heavy_deps() -> None:
    """The serving image installs neither ultralytics nor torch nor pillow-at-import.

    cv2 (headless) is the one image dependency the module is allowed — the shared
    decode + resize implementation that makes parity with ultralytics hold by
    construction.
    """
    check = (
        "import sys; import mlops_cv.serving.prepost; "
        "banned = {'torch', 'ultralytics', 'PIL'} & set(sys.modules); "
        "assert not banned, banned"
    )
    subprocess.run([sys.executable, "-c", check], check=True)
