# GPU setup

An NVIDIA GPU must work in two independent places: **host Python** (torch built for your GPU) and
**containers** (via the NVIDIA Container Toolkit).

> Validated on: RTX 5070 (Blackwell `sm_120`, 8 GB), driver 595.71.05 / CUDA 13.2, Fedora 44,
> **rootful** Docker. Values tied to that box are noted below.

## Host torch

```bash
make setup
make gpu-smoke    # proves the installed torch actually runs on your GPU
```

`gpu-smoke` runs a real GPU matmul (not just `torch.cuda.is_available()`, which a CPU-fallback
wheel also passes) and prints e.g. `OK: real GPU matmul correct on … (sm_120)`.

torch comes from a CUDA-matched wheel index, not PyPI — wired in [pyproject.toml](../pyproject.toml).
For a different GPU, point that index at the `cuXXX` matching your driver's CUDA (`nvidia-smi`) and re-pin torch/torchvision.

## Containers — NVIDIA Container Toolkit

```bash
# Add NVIDIA's repo, install, register with Docker (Fedora example).
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit-1.19.1
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

Validate the boundary with the provided base image
([docker/Dockerfile.base](../docker/Dockerfile.base); its tag should match your driver's CUDA):

```bash
docker build -f docker/Dockerfile.base -t mlops-cv-base .
docker run --rm --gpus all mlops-cv-base        # prints nvidia-smi from inside the container
```
