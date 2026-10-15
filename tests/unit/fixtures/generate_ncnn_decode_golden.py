"""Regenerate ``ncnn_decode_golden.npz``: a real NCNN ``out0`` and ultralytics' NMS of it.

CI pins ``mlops_cv.edge.runtime.decode`` against it without ultralytics or ncnn. Run it
locally on the champion's ``ncnn-fp16-320`` and one test frame, only when ultralytics
changes ``non_max_suppression``:

    uv run python tests/unit/fixtures/generate_ncnn_decode_golden.py \
        models/ncnn-fp16-320_ncnn_model <subset>/images/test/uav0000355_00001_v_0000181.jpg

The goldens come from ultralytics, never from the module, and the script refuses to write
a fixture ``decode`` disagrees with.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).parent / "ncnn_decode_golden.npz"
IMGSZ = 320
# name -> (conf, multi_label): predict's defaults, and val's.
SETTINGS = {"predict": (0.25, False), "val": (0.001, True)}
IOU, MAX_DET = 0.7, 300


def main(model_dir: Path, frame: Path) -> None:
    import torch
    from ultralytics.utils.nms import non_max_suppression

    from mlops_cv.edge.runtime import NcnnDetector, decode
    from mlops_cv.serving import prepost

    detector = NcnnDetector(model_dir, IMGSZ)
    out0 = detector.forward(prepost.preprocess(prepost.load_image(frame), IMGSZ))
    payload = {"out0": out0}
    for name, (conf, multi_label) in SETTINGS.items():
        (golden,) = non_max_suppression(
            torch.from_numpy(out0[None].copy()),
            conf,
            IOU,
            multi_label=multi_label,
            max_det=MAX_DET,
        )
        golden = golden.numpy()
        ours = decode(out0, conf, IOU, max_det=MAX_DET, multi_label=multi_label)
        assert np.array_equal(ours, golden), f"decode diverges from ultralytics: {name}"
        payload[name] = golden
        print(f"{name}: {len(golden)} detections")

    np.savez_compressed(FIXTURE, **payload)  # type: ignore
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
