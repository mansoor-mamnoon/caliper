"""corpus:attention_fwd -- FlashAttention-style forward: a Triton kernel + the
``F.scaled_dot_product_attention`` (SDPA) baseline.

One Triton program per (batch, head, query block), streaming K/V blocks with
the online-softmax running max / sum (FlashAttention-2 shape -- this kernel
follows that structure, it is not copied from any upstream file). Supports a
causal mask and grouped-query attention (``h_kv`` < ``h`` query heads share a
K/V head); head dim 64 or 128; bf16 / fp16 / fp32. Both the kernel and the
baseline (forward only, on the same expanded K/V) run through
:func:`caliper.live_timing_ms`; see ``caliper.corpus._common`` for why.

Importable without Triton or PyTorch; :func:`run` needs both, plus a CUDA
device.
"""

from __future__ import annotations

from typing import Any

from caliper import Result
from caliper.corpus._common import (
    assemble_result,
    attention_dims,
    attention_torch_dtype,
    content_hash,
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
    "CONFIG",
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

#: (BLOCK_M, BLOCK_N) query/key tile. A single default -- attention autotuning
#: is out of scope for the corpus (gemm carries the autotune contract) -- but a
#: caller's ``run(cell, config)`` override is applied and recorded.
CONFIG = {"BLOCK_M": 64, "BLOCK_N": 64}

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
            qk = tl.dot(q, k, allow_tf32=False) * sm_scale
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
            acc += tl.dot(p, v, allow_tf32=False)
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


def _resolve(config: dict[str, int] | None) -> dict[str, int]:
    return {**CONFIG, **(config or {})}


def _make_qkv(b: int, h: int, s: int, d: int, h_kv: int, torch_dtype: Any) -> tuple[Any, Any, Any]:
    """Random Q ``(b, h, s, d)`` and K/V ``(b, h_kv, s, d)`` on the device."""
    import torch

    q = torch.randn((b, h, s, d), device="cuda", dtype=torch_dtype)
    k = torch.randn((b, h_kv, s, d), device="cuda", dtype=torch_dtype)
    v = torch.randn((b, h_kv, s, d), device="cuda", dtype=torch_dtype)
    return q, k, v


def _expand(k: Any, v: Any, q_heads: int) -> tuple[Any, Any]:
    """K/V repeated to ``q_heads`` heads (version-independent GQA), materialised
    once so it is not redone inside a timed loop."""
    group = q_heads // k.shape[1]
    return k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)


def _sdpa(q: Any, k_full: Any, v_full: Any, causal: bool) -> Any:
    import torch.nn.functional as F

    return F.scaled_dot_product_attention(q, k_full, v_full, is_causal=causal)


def _launch_triton(q: Any, k: Any, v: Any, causal: bool, cfg: dict[str, int]) -> Any:
    import torch

    b, h, s, d = q.shape
    h_kv = k.shape[1]
    out = torch.empty_like(q)
    lse = torch.empty((b * h, s), device="cuda", dtype=torch.float32)
    sm_scale = 1.0 / (d**0.5)
    block_m, block_n = cfg["BLOCK_M"], cfg["BLOCK_N"]
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


def check_numerics(cell: dict[str, Any], config: dict[str, int] | None = None) -> dict[str, Any]:
    """Run the Triton kernel and the SDPA baseline once (untimed) and compare.
    Returns ``{"max_abs_err", "max_rel_err", "allclose"}``; the DoD's
    ``allclose`` check. Needs Triton, PyTorch, and a CUDA device."""
    require_live_deps("attention_fwd")
    import torch

    b, h, s, d, h_kv, causal = attention_dims(cell)
    torch_dtype = attention_torch_dtype(cell.get("dtype", "bf16"))
    q, k, v = _make_qkv(b, h, s, d, h_kv, torch_dtype)
    k_full, v_full = _expand(k, v, h)

    got = _launch_triton(q, k, v, causal, _resolve(config)).to(torch.float32)
    want = _sdpa(q, k_full, v_full, causal).to(torch.float32)
    abs_err = (got - want).abs()
    tol = 3e-3 if torch_dtype == torch.float32 else 2e-2
    return {
        "max_abs_err": float(abs_err.max()),
        "max_rel_err": float((abs_err / want.abs().clamp_min(1e-6)).max()),
        "allclose": torch.allclose(got, want, atol=tol, rtol=tol),
    }


def run(cell: dict[str, Any], config: dict[str, int] | None = None) -> Result:
    """Time this attention forward at ``cell``'s shape/dtype (``cell`` may also
    carry ``causal`` / ``h_kv``), against the SDPA baseline on the same
    expanded K/V. Needs Triton, PyTorch, and a CUDA device."""
    require_live_deps("attention_fwd")

    b, h, s, d, h_kv, causal = attention_dims(cell)
    dtype_name = cell.get("dtype", "bf16")
    torch_dtype = attention_torch_dtype(dtype_name)
    cfg = _resolve(config)
    q, k, v = _make_qkv(b, h, s, d, h_kv, torch_dtype)
    k_full, v_full = _expand(k, v, h)  # once, outside the timed loop

    samples_us = [t * 1000.0 for t in time_kernel(lambda: _launch_triton(q, k, v, causal, cfg))]
    baseline_us = [t * 1000.0 for t in time_kernel(lambda: _sdpa(q, k_full, v_full, causal))]
    triton_p50 = sorted(samples_us)[len(samples_us) // 2]
    baseline_p50 = sorted(baseline_us)[len(baseline_us) // 2]

    spec = roofline_spec({"B": b, "H": h, "S": s, "D": d, "causal": causal}, dtype_name)

    return assemble_result(
        kernel_name="attention_fwd",
        kernel_impl="triton",
        dtype=dtype_name,
        layout=None,
        shape={"B": b, "H": h, "S": s, "D": d, "h_kv": h_kv, "causal": causal},
        source_hash=SOURCE_HASH,
        autotune_config=cfg,
        samples_us=samples_us,
        machine=torch_machine(),
        flops=spec["flops"] if spec else None,
        bytes_hbm=spec["bytes_hbm"] if spec else None,
        baseline="sdpa",
        baseline_pct=(baseline_p50 / triton_p50) if triton_p50 > 0 else None,
    )
