"""Unit tests for slicing train labels into SAHI windows (no images, no SAHI)."""

from __future__ import annotations

import numpy as np

from mlops_cv.data.slices import slice_labels


def test_boxes_are_renormalised_clipped_and_dropped_per_window() -> None:
    # A 200x100 frame: a box inside the left window, one straddling both, one 5% inside the right.
    labels = np.array(
        [
            [0, 0.25, 0.5, 0.1, 0.2],  # x 40-60
            [1, 0.5, 0.5, 0.2, 0.2],  # x 80-120
            [2, 0.4, 0.5, 0.2, 0.2],  # x 60-100: 5% of its width past x=99
        ]
    )
    left = slice_labels(labels, (200, 100), (0, 0, 100, 100))
    np.testing.assert_allclose(
        left, [[0, 0.5, 0.5, 0.2, 0.2], [1, 0.9, 0.5, 0.2, 0.2], [2, 0.8, 0.5, 0.4, 0.2]]
    )
    right = slice_labels(labels, (200, 100), (99, 0, 199, 100))
    np.testing.assert_allclose(right, [[1, 0.105, 0.5, 0.21, 0.2]], atol=1e-9)
    assert slice_labels(np.array([]), (200, 100), (0, 0, 100, 100)).shape == (0, 5)
