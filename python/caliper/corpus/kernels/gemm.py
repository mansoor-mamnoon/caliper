"""corpus:gemm -- dense GEMM: a Triton kernel + the cuBLAS baseline.

``(M, K) x (K, N) -> (M, N)``. The Triton kernel is block-tiled with grouped
column ordering for L2 reuse (the standard shape for a matmul kernel); the
baseline is ``torch.matmul`` (cuBLAS). Both run through
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

__all__ = ["CONFIGS", "KERNEL_KEY", "SOURCE_HASH", "kernel", "roofline_spec", "run"]

#: matches ``corpus:gemm`` -> ``corpus:gemm_bf16`` in crates/caliper-gpu/src/corpus.rs
KERNEL_KEY = "corpus:gemm_bf16"
SOURCE_HASH = content_hash(__file__)

#: block-tiling configs to autotune over. ``num_warps`` / ``num_stages`` are
#: fixed per config below; a ``run()`` config dict carries only these keys.
CONFIGS: list[dict[str, int]] = [
    {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
    {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
    {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
]
_NUM_WARPS_STAGES: dict[tuple[int, int, int, int], tuple[int, int]] = {
    (64, 64, 32, 8): (4, 4),
    (128, 64, 32, 8): (4, 4),
    (64, 128, 32, 8): (4, 4),
    (128, 128, 32, 8): (8, 3),
    (128, 256, 64, 8): (8, 3),
}


def _warps_stages(cfg: dict[str, int]) -> tuple[int, int]:
    key = (cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"], cfg.get("GROUP_M", 8))
    return _NUM_WARPS_STAGES.get(key, (4, 3))


if TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _kernel(
        a_ptr: Any,
        b_ptr: Any,
        c_ptr: Any,
        M: Any,
        N: Any,
        K: Any,
        stride_am: Any,
        stride_ak: Any,
        stride_bk: Any,
        stride_bn: Any,
        stride_cm: Any,
        stride_cn: Any,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ) -> None:
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a_mask = (rm[:, None] < M) & (rk[None, :] < K - k * BLOCK_K)
            b_mask = (rk[:, None] < K - k * BLOCK_K) & (rn[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)

    kernel = triton.autotune(
        configs=[
            triton.Config(cfg, num_warps=w, num_stages=s)
            for cfg, (w, s) in ((c, _warps_stages(c)) for c in CONFIGS)
        ],
        key=["M", "N", "K"],
    )(_kernel)
else:  # pragma: no cover - triton not installed on the dev box
    _kernel = None
    kernel = None


def roofline_spec(shape: dict[str, Any], dtype: str) -> dict[str, Any] | None:
    """The FLOP / HBM-byte roofline spec for ``shape={"M","N","K"}`` at
    ``dtype``, or ``None`` if a dimension is missing. Pure; no GPU needed."""
    spec_json = corpus_roofline_spec(KERNEL_KEY, json.dumps(shape), dtype)
    return dict(json.loads(spec_json)) if spec_json is not None else None


def _dim(shape: dict[str, Any], *names: str) -> int:
    for name in names:
        if name in shape:
            return int(shape[name])
    raise KeyError(f"shape is missing one of {names}: {shape!r}")


def run(cell: dict[str, Any], config: dict[str, int] | None = None) -> Result:
    """Time this GEMM at ``cell``'s shape/dtype/layout with one autotune
    ``config`` (default: the first entry of :data:`CONFIGS`), against the
    cuBLAS baseline. Needs Triton, PyTorch, and a CUDA device."""
    require_live_deps("gemm")
    import torch

    shape = cell["shape"]
    m, n, k = _dim(shape, "m", "M"), _dim(shape, "n", "N"), _dim(shape, "k", "K")
    dtype_name = cell.get("dtype", "bf16")
    layout = cell.get("layout", "row")
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    cfg = dict(config) if config else CONFIGS[0]

    a = torch.randn((m, k), device="cuda", dtype=torch_dtype)
    b_rows_major = torch.randn((k, n), device="cuda", dtype=torch_dtype)
    b = b_rows_major.t().contiguous().t() if layout == "col" else b_rows_major
    c = torch.empty((m, n), device="cuda", dtype=torch_dtype)

    block_m, block_n, block_k = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]
    group_m = cfg.get("GROUP_M", 8)
    num_warps, num_stages = _warps_stages(cfg)

    def grid(_meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    def launch() -> None:
        assert _kernel is not None
        _kernel[grid](
            a,
            b,
            c,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    samples_us = [t * 1000.0 for t in time_kernel(launch)]
    baseline_us = [t * 1000.0 for t in time_kernel(lambda: torch.matmul(a, b, out=c))]
    triton_p50 = sorted(samples_us)[len(samples_us) // 2]
    baseline_p50 = sorted(baseline_us)[len(baseline_us) // 2]

    spec = roofline_spec({"M": m, "N": n, "K": k}, dtype_name)

    return assemble_result(
        kernel_name="gemm",
        kernel_impl="triton",
        dtype=dtype_name,
        layout=layout,
        shape={"M": m, "N": n, "K": k},
        source_hash=SOURCE_HASH,
        autotune_config=cfg,
        samples_us=samples_us,
        flops=spec["flops"] if spec else None,
        bytes_hbm=spec["bytes_hbm"] if spec else None,
        baseline="cublas",
        baseline_pct=(baseline_p50 / triton_p50) if triton_p50 > 0 else None,
    )
