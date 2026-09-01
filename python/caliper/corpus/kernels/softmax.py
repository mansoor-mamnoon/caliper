"""corpus:softmax -- row-wise softmax forward: a Triton kernel + a torch
baseline.

``y = exp(x - max(x, axis=-1)) / sum(exp(x - max(x, axis=-1)), axis=-1)``, one
program per row. The baseline is ``torch.softmax``. Both run through
:func:`caliper.live_timing_ms`; see ``caliper.corpus._common`` for why.

Importable without Triton or PyTorch; :func:`run` needs both, plus a CUDA
device.
"""

from __future__ import annotations

import json
from typing import Any

from caliper import Result
from caliper._core import corpus_roofline_spec
from caliper.corpus._common import assemble_result, content_hash, require_live_deps, time_kernel

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - triton not installed on the dev box
    TRITON_AVAILABLE = False

__all__ = ["KERNEL_KEY", "SOURCE_HASH", "kernel", "roofline_spec", "run"]

#: matches ``corpus:softmax`` in crates/caliper-gpu/src/corpus.rs
KERNEL_KEY = "corpus:softmax"
SOURCE_HASH = content_hash(__file__)

if TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def kernel(
        x_ptr: Any,
        y_ptr: Any,
        n_cols: Any,
        stride_row: Any,
        BLOCK_SIZE: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        numerator = tl.exp(x)
        y = numerator / tl.sum(numerator, axis=0)
        tl.store(y_ptr + row * stride_row + cols, y, mask=mask)
else:  # pragma: no cover - triton not installed on the dev box
    kernel = None


def roofline_spec(shape: dict[str, Any], dtype: str) -> dict[str, Any] | None:
    """The FLOP / HBM-byte roofline spec for ``shape={"ROWS","COLS"}`` at
    ``dtype``, or ``None`` if a dimension is missing. Pure; no GPU needed."""
    spec_json = corpus_roofline_spec(KERNEL_KEY, json.dumps(shape), dtype)
    return dict(json.loads(spec_json)) if spec_json is not None else None


def _dim(shape: dict[str, Any], *names: str) -> int:
    for name in names:
        if name in shape:
            return int(shape[name])
    raise KeyError(f"shape is missing one of {names}: {shape!r}")


def run(cell: dict[str, Any], config: dict[str, int] | None = None) -> Result:
    """Time softmax forward at ``cell``'s shape/dtype, against the torch
    baseline. Needs Triton, PyTorch, and a CUDA device."""
    require_live_deps("softmax")
    import torch

    shape = cell["shape"]
    rows, cols = _dim(shape, "rows", "ROWS"), _dim(shape, "cols", "COLS")
    dtype_name = cell.get("dtype", "bf16")
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    block_size = triton.next_power_of_2(cols)

    x = torch.randn((rows, cols), device="cuda", dtype=torch_dtype)
    y = torch.empty_like(x)

    def launch() -> None:
        assert kernel is not None
        kernel[(rows,)](x, y, cols, x.stride(0), BLOCK_SIZE=block_size)

    samples_us = [t * 1000.0 for t in time_kernel(launch)]
    baseline_us = [t * 1000.0 for t in time_kernel(lambda: torch.softmax(x, dim=-1))]
    triton_p50 = sorted(samples_us)[len(samples_us) // 2]
    baseline_p50 = sorted(baseline_us)[len(baseline_us) // 2]

    spec = roofline_spec({"ROWS": rows, "COLS": cols}, dtype_name)

    return assemble_result(
        kernel_name="softmax",
        kernel_impl="triton",
        dtype=dtype_name,
        layout=None,
        shape={"ROWS": rows, "COLS": cols},
        source_hash=SOURCE_HASH,
        autotune_config=dict(config or {"BLOCK_SIZE": block_size}),
        samples_us=samples_us,
        flops=spec["flops"] if spec else None,
        bytes_hbm=spec["bytes_hbm"] if spec else None,
        baseline="torch",
        baseline_pct=(baseline_p50 / triton_p50) if triton_p50 > 0 else None,
    )
