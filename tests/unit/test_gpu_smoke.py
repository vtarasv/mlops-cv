"""GPU smoke tests"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu


def test_cuda_available() -> None:
    assert torch.cuda.is_available(), "no CUDA device / CPU-only torch build"


def test_real_gpu_matmul_matches_cpu() -> None:
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    torch.testing.assert_close((a @ b).cpu(), a.cpu() @ b.cpu(), rtol=1e-3, atol=1e-3)
