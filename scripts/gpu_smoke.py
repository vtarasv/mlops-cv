"""GPU smoke check — verifies the installed PyTorch can actually run on the GPU present.

Architecture-agnostic: it detects the device's compute capability and runs a real matmul whose
result must match the CPU. That is what catches the failures `torch.cuda.is_available()` misses.
"""

from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"torch {torch.__version__}")

    if not torch.cuda.is_available():
        print(
            "FAIL: CUDA not available — likely a CPU-only torch build, or a driver/toolkit "
            "mismatch. Install torch from the CUDA wheel index that matches your GPU + driver.",
            file=sys.stderr,
        )
        return 1

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    device_arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    print(f"device: {name}  (compute capability {device_arch})")
    print(f"wheel arch list: {arch_list}")

    if device_arch not in arch_list:
        print(
            f"WARNING: this wheel has no native {device_arch} kernels; relying on PTX JIT. "
            "If the next step fails, install torch wheels built for your GPU's CUDA arch.",
            file=sys.stderr,
        )

    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    out = a @ b
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(), a.cpu() @ b.cpu(), rtol=1e-3, atol=1e-3)

    print(f"OK: real GPU matmul correct on {name} ({device_arch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
