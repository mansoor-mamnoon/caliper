"""Reference kernel implementations: ``gemm``, ``rmsnorm``, ``softmax``.

Each submodule is importable without Triton or PyTorch installed (only
``.run()`` needs them, and a CUDA device); see ``caliper.corpus._common``.
"""

from __future__ import annotations

__all__: list[str] = []
