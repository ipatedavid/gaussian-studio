"""The pod's PyTorch environment, shared by the steps that run AI models in Python.

People masking needs it on its own; multi-view depth guidance (when present) adds its
packages to the same environment, so PyTorch is installed once per pod.

Every function takes the running pipeline module as `worker`, so it uses the
pipeline's own run_command and downloads (and whatever tests patch in).
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

ENV = Path("/workspace/torchenv")
UV_DIR = Path("/workspace/uv")
UV_VERSION = "0.13.0"
BASE_PACKAGES = ("numpy<2", "opencv-python-headless", "pillow")
# PyTorch's CUDA 12.6 builds stop at Hopper; Blackwell cards (RTX 50xx, B200: compute
# capability 10 and up) need the CUDA 12.8 builds.
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
TORCH_INDEX_BLACKWELL = "https://download.pytorch.org/whl/cu128"


def torch_index() -> str:
    try:
        caps = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.split()
        major = int(float(caps[0]))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return TORCH_INDEX
    return TORCH_INDEX_BLACKWELL if major >= 10 else TORCH_INDEX


def uv(worker) -> str:
    """The uv installer, downloaded once per pod when the image has none."""
    found = shutil.which("uv")
    if found:
        return found
    binary = UV_DIR / "uv"
    if binary.is_file():
        return str(binary)
    UV_DIR.mkdir(parents=True, exist_ok=True)
    archive = UV_DIR / "uv.tar.gz"
    worker._download_file(
        f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/"
        "uv-x86_64-unknown-linux-gnu.tar.gz", archive, "uv")
    # --no-same-owner: some pod volumes refuse chown, even for root.
    worker.run_command(["tar", "-xzf", str(archive), "-C", str(UV_DIR),
                        "--strip-components=1", "--no-same-owner"])
    return str(binary)


def ensure(worker, env: Path | None = None) -> Path:
    """Python with torch, torchvision, NumPy and OpenCV in env; returns its interpreter."""
    env = env or ENV
    python = env / "bin" / "python"
    ready = env / "torch-ready"
    if ready.is_file():
        return python
    started = time.time()
    installer = uv(worker)
    print("[pipeline] setting up PyTorch for CUDA (once per pod)", flush=True)
    worker.run_command([installer, "venv", "--python", "3.12", str(env)])
    worker.run_command([installer, "pip", "install", "--python", str(python),
                        "--index-url", torch_index(), "torch", "torchvision"])
    worker.run_command([installer, "pip", "install", "--python", str(python), *BASE_PACKAGES])
    ready.write_text("ok")
    print(f"[pipeline] PyTorch ready in {time.time() - started:.0f}s", flush=True)
    return python
