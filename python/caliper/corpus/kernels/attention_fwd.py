"""corpus:attention_fwd -- FlashAttention-style forward: a Triton kernel + the
``F.scaled_dot_product_attention`` (SDPA) baseline.

One Triton program per (batch, head, query block), streaming K/V blocks with
the online-softmax running max / sum (FlashAttention-2 shape). Supports a
causal mask and grouped-query attention (``h_kv`` < ``h`` query heads share a
K/V head); head dim 64 or 128; bf16 / fp16 / fp32. Both the kernel and the
baseline run through :func:`caliper.live_timing_ms`; see
``caliper.corpus._common`` for why.

Importable without Triton or PyTorch; :func:`run` needs both, plus a CUDA
device.
"""

from __future__ import annotations

from typing import Any

from caliper import Result
from caliper.corpus._common import (
    assemble_result,
    content_hash,
    dim,
    require_live_deps,
    roofline_spec_for,
    time_kernel,
    torch_machine,
)

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - triton not installed on the dev box
    TRITON_AVAILABLE = False

__all__ = [
    "KERNEL_KEY",
    "SOURCE_HASH",
    "check_numerics",
    "kernel",
    "roofline_spec",
    "run",
]

#: matches ``corpus:attention_fwd`` in crates/caliper-gpu/src/corpus.rs
KERNEL_KEY = "corpus:attention_fwd"
SOURCE_HASH = content_hash(__file__)

#: (BLOCK_M, BLOCK_N) query/key tile; a single entry -- attention autotuning is
#: out of scope for the corpus (gemm carries the autotune contract).
CONFIG = {"BLOCK_M": 64, "BLOCK_N": 64}
_TORCH_DTYPES = ("bf16", "fp16", "fp32")

if TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def kernel(
        q_ptr: Any,
        k_ptr: Any,
        v_ptr: Any,
        out_ptr: Any,
        lse_ptr: Any,
        sm_scale: Any,
        stride_qb: Any,
        stride_qh: Any,
        stride_qm: Any,
        stride_qd: Any,
        stride_kb: Any,
        stride_kh: Any,
        stride_kn: Any,
        stride_kd: Any,
        stride_vb: Any,
        stride_vh: Any,
        stride_vn: Any,
        stride_vd: Any,
        stride_ob: Any,
        stride_oh: Any,
        stride_om: Any,
        stride_od: Any,
        n_ctx: Any,
        q_heads: Any,
        kv_heads: Any,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
    ) -> None:
        start_m = tl.program_id(0)
        off_bh = tl.program_id(1)
        off_b = off_bh // q_heads
        off_h = off_bh % q_heads
        off_hkv = off_h // (q_heads // kv_heads)  # grouped-query attention

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
        k_base = k_ptr + off_b * stride_kb + off_hkv * stride_kh
        v_base = v_ptr + off_b * stride_vb + off_hkv * stride_vh

        q = tl.load(
            q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=offs_m[:, None] < n_ctx,
            other=0.0,
        ).to(tl.float32)

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

        hi = (start_m + 1) * BLOCK_M if CAUSAL else n_ctx
        for start_n in range(0, hi, BLOCK_N):
            n_idx = start_n + offs_n
            k = tl.load(
                k_base + n_idx[None, :] * stride_kn + offs_d[:, None] * stride_kd,
                mask=n_idx[None, :] < n_ctx,
                other=0.0,
            ).to(tl.float32)
            qk = tl.dot(q, k) * sm_scale
            qk += tl.where(n_idx[None, :] < n_ctx, 0.0, float("-inf"))
            if CAUSAL:
                qk += tl.where(offs_m[:, None] >= n_idx[None, :], 0.0, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            v = tl.load(
                v_base + n_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=n_idx[:, None] < n_ctx,
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(p, v)
            m_i = m_ij

        acc = acc / l_i[:, None]
        tl.store(lse_ptr + off_bh * n_ctx + offs_m, m_i + tl.log(l_i), mask=offs_m < n_ctx)
        tl.store(
            out_ptr
            + off_b * stride_ob
            + off_h * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_od,
            acc.to(out_ptr.dtype.element_ty),
            mask=offs_m[:, None] < n_ctx,
        )
else:  # pragma: no cover - triton not installed on the dev box
    kernel = None


def roofline_spec(shape: dict[str, Any], dtype: str) -> dict[str, Any] | None:
    """The FLOP / HBM-byte roofline spec for ``shape={"B","H","S","D"}`` (plus
    an optional ``"causal"`` bool) at ``dtype``, or ``None`` if a dimension is
    missing. Pure; no GPU needed."""
    return roofline_spec_for(KERNEL_KEY, shape, dtype)


def _dims(cell: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    """``(b, h, s, d, h_kv, causal)`` from a cell. ``h_kv`` defaults to ``h``
    (plain multi-head); ``causal`` defaults to ``False``."""
    shape = cell["shape"]
    b, h, s, d = (
        dim(shape, "b", "B"),
        dim(shape, "h", "H"),
        dim(shape, "s", "S"),
        dim(shape, "d", "D"),
    )
    h_kv = int(cell.get("h_kv") or shape.get("h_kv") or h)
    if h % h_kv != 0:
        raise ValueError(f"h={h} must be a multiple of h_kv={h_kv}")
    causal = bool(cell.get("causal") or shape.get("causal") or False)
    return b, h, s, d, h_kv, causal


def _make_qkv(b: int, h: int, s: int, d: int, h_kv: int, torch_dtype: Any) -> tuple[Any, Any, Any]:
    import torch

    q = torch.randn((b, h, s, d), device="cuda", dtype=torch_dtype)
    k = torch.randn((b, h_kv, s, d), device="cuda", dtype=torch_dtype)
    v = torch.randn((b, h_kv, s, d), device="cuda", dtype=torch_dtype)
    return q, k, v


def _sdpa_baseline(q: Any, k: Any, v: Any, causal: bool) -> Any:
    """``F.scaled_dot_product_attention`` with K/V expanded to the query head
    count (version-independent GQA)."""
    import torch.nn.functional as F

    group = q.shape[1] // k.shape[1]
    k_full = k.repeat_interleave(group, dim=1)
    v_full = v.repeat_interleave(group, dim=1)
    return F.scaled_dot_product_attention(q, k_full, v_full, is_causal=causal)


def _launch_triton(q: Any, k: Any, v: Any, causal: bool) -> Any:
    import torch

    b, h, s, d = q.shape
    h_kv = k.shape[1]
    out = torch.empty_like(q)
    lse = torch.empty((b * h, s), device="cuda", dtype=torch.float32)
    sm_scale = 1.0 / (d**0.5)
    block_m, block_n = CONFIG["BLOCK_M"], CONFIG["BLOCK_N"]
    grid = (triton.cdiv(s, block_m), b * h)
    assert kernel is not None
    kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        s,
        h,
        h_kv,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=d,
        CAUSAL=causal,
    )
    return out


def _torch_dtype(dtype_name: str) -> Any:
    import torch

    if dtype_name not in _TORCH_DTYPES:
        raise NotImplementedError(
            f"corpus:attention_fwd supports {_TORCH_DTYPES}; {dtype_name!r} "
            "(e.g. the fp8 path for L4) is a follow-up."
        )
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]


def check_numerics(cell: dict[str, Any], config: dict[str, int] | None = None) -> dict[str, Any]:
    """Run the Triton kernel and the SDPA baseline once (untimed) and compare.
    Returns ``{"max_abs_err", "max_rel_err", "allclose"}``; the DoD's
    ``allclose`` check. Needs Triton, PyTorch, and a CUDA device."""
    require_live_deps("attention_fwd")
    import torch

    b, h, s, d, h_kv, causal = _dims(cell)
    torch_dtype = _torch_dtype(cell.get("dtype", "bf16"))
    q, k, v = _make_qkv(b, h, s, d, h_kv, torch_dtype)

    got = _launch_triton(q, k, v, causal).to(torch.float32)
    want = _sdpa_baseline(q, k, v, causal).to(torch.float32)
    abs_err = (got - want).abs()
    max_abs = float(abs_err.max())
    max_rel = float((abs_err / want.abs().clamp_min(1e-6)).max())
    tol = 3e-3 if torch_dtype == torch.float32 else 2e-2
    return {
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "allclose": torch.allclose(got, want, atol=tol, rtol=tol),
    }


def run(cell: dict[str, Any], config: dict[str, int] | None = None) -> Result:
    """Time this attention forward at ``cell``'s shape/dtype (``cell`` may also
    carry ``causal`` / ``h_kv``), against the SDPA baseline. Needs Triton,
    PyTorch, and a CUDA device."""
    require_live_deps("attention_fwd")

    b, h, s, d, h_kv, causal = _dims(cell)
    dtype_name = cell.get("dtype", "bf16")
    torch_dtype = _torch_dtype(dtype_name)
    q, k, v = _make_qkv(b, h, s, d, h_kv, torch_dtype)

    samples_us = [t * 1000.0 for t in time_kernel(lambda: _launch_triton(q, k, v, causal))]
    baseline_us = [t * 1000.0 for t in time_kernel(lambda: _sdpa_baseline(q, k, v, causal))]
    triton_p50 = sorted(samples_us)[len(samples_us) // 2]
    baseline_p50 = sorted(baseline_us)[len(baseline_us) // 2]

    spec_shape = {"B": b, "H": h, "S": s, "D": d, "causal": causal}
    spec = roofline_spec(spec_shape, dtype_name)

    return assemble_result(
        kernel_name="attention_fwd",
        kernel_impl="triton",
        dtype=dtype_name,
        layout=None,
        shape={"B": b, "H": h, "S": s, "D": d, "h_kv": h_kv, "causal": causal},
        source_hash=SOURCE_HASH,
        autotune_config=dict(config or CONFIG),
        samples_us=samples_us,
        machine=torch_machine(),
        flops=spec["flops"] if spec else None,
        bytes_hbm=spec["bytes_hbm"] if spec else None,
        baseline="sdpa",
        baseline_pct=(baseline_p50 / triton_p50) if triton_p50 > 0 else None,
    )
