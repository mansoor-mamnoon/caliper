"""Detect the local kernel toolchain: Triton, PyTorch, and the CUDA compilers.

The subprocess calls and package lookups belong on the Python side; the version
strings themselves are parsed by the Rust core (``caliper._core``) so the format
handling is shared with the rest of the schema.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from importlib import metadata

from caliper import _core

__all__ = ["detect"]

# `nvcc --version` can be slow to load on older toolkits; keep the wait short.
_TIMEOUT_S = 10.0


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _tool_version(exe: str, parse: Callable[[str], str | None]) -> str | None:
    """Run ``<exe> --version`` and hand its output to the Rust parser ``parse``."""
    path = shutil.which(exe)
    if path is None:
        return None
    try:
        proc = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse(proc.stdout or proc.stderr or "")


def _torch_cuda_runtime() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    version = getattr(getattr(torch, "version", None), "cuda", None)
    return version if isinstance(version, str) else None


def detect() -> dict[str, str | None]:
    """The toolchain as a dict shaped like the schema ``machine.toolkit`` block.

    Every value is the detected version string or ``None`` when the tool or
    package is not installed. ``cuda_runtime`` is PyTorch's bundled CUDA version
    when torch is importable, else the ``nvcc`` version.
    """
    nvcc = _tool_version("nvcc", _core.parse_nvcc_version)
    ptxas = _tool_version("ptxas", _core.parse_ptxas_version)
    return {
        "triton": _package_version("triton"),
        "torch": _package_version("torch"),
        "ptxas": ptxas,
        "nvcc": nvcc,
        "cuda_runtime": _torch_cuda_runtime() or nvcc,
    }
