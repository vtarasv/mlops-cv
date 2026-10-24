"""Unit tests for the SAHI evaluation's pure helpers (no GPU, no SAHI)."""

from __future__ import annotations

import numpy as np

from mlops_cv.tuning.sahi_eval import groups, resolution_group, top_k


def test_resolution_groups_split_val_by_source_width() -> None:
    assert [resolution_group(w) for w in (3840, 2720, 2688, 1904, 1344)] == [
        "3840px",
        "2720px",
        "2720px",
        "1904px",
        "1904px",
    ]
    sizes = {"a.jpg": (3840, 2160), "b.jpg": (1344, 756), "c.jpg": (1904, 1071)}
    assert groups(sizes) == {"1904px": ["b.jpg", "c.jpg"], "3840px": ["a.jpg"]}


def test_top_k_keeps_the_highest_confidence_rows_in_order() -> None:
    rows = np.array([[0, 0, 1, 1, c, 0] for c in (0.1, 0.9, 0.5, 0.7)])
    assert top_k(rows, k=2)[:, 4].tolist() == [0.9, 0.7]
    assert top_k(np.zeros((0, 6))).shape == (0, 6)
