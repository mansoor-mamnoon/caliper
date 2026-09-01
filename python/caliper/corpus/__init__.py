"""The reference kernel corpus: Triton implementations + vendor baselines.

Each kernel in :mod:`caliper.corpus.kernels` pairs a Triton kernel with a
vendor baseline (cuBLAS via ``torch.matmul``, or the relevant ``torch``
op) and a pure, no-GPU-needed roofline model. See
``docs/corpus.md`` for what each kernel computes and how it is pinned.
"""

from __future__ import annotations

__all__: list[str] = []
