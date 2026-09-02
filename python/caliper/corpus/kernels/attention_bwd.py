"""corpus:attention_bwd -- FlashAttention-style backward: a Triton kernel + the
SDPA autograd baseline.

The backward pass for scaled-dot-product attention (this kernel follows the
FlashAttention-2 backward structure, it is not copied from any upstream file).
A preprocess kernel forms ``delta = rowsum(dO * O)``; the main kernel takes one
K/V block per program, sweeps the query blocks, accumulates ``dK`` / ``dV`` in
registers, and adds into ``dQ`` with atomics (``dQ`` is an fp32 scratch
buffer, cast down on return). Causal supported; grouped-query attention is
handled by expanding K/V to the query head count and summing each group's
``dK`` / ``dV`` back down afterward. bf16 / fp16 / fp32; head dim 64 or 128.

Both sides are timed backward-only: the Triton path from given
``Q, K, V, O, LSE, dO``; the baseline as ``torch.autograd.grad`` over a single
(untimed) SDPA forward. Both go through :func:`caliper.live_timing_ms`; see
``caliper.corpus._common`` for why.

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

#: matches ``corpus:attention_bwd`` in crates/caliper-gpu/src/corpus.rs
KERNEL_KEY = "corpus:attention_bwd"
SOURCE_HASH = content_hash(__file__)

#: (BLOCK_M, BLOCK_N) tile default; a caller's ``run(cell, config)`` override
#: is applied and recorded (see ``attention_fwd.CONFIG``).
CONFIG = {"BLOCK_M": 64, "BLOCK_N": 64}

if TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _preprocess(
        o_ptr: Any,
        do_ptr: Any,
        delta_ptr: Any,
        stride_ob: Any,
        stride_oh: Any,
        stride_om: Any,
        stride_od: Any,
        n_ctx: Any,
        q_heads: Any,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        start_m = tl.program_id(0)
        off_bh = tl.program_id(1)
        off_b = off_bh // q_heads
        off_h = off_bh % q_heads
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        base = off_b * stride_ob + off_h * stride_oh
        ptrs = base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        mask = offs_m[:, None] < n_ctx
        o = tl.load(o_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        do = tl.load(do_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(delta_ptr + off_bh * n_ctx + offs_m, tl.sum(o * do, axis=1), mask=offs_m < n_ctx)

    @triton.jit  # type: ignore[untyped-decorator]
    def kernel(
        q_ptr: Any,
        k_ptr: Any,
        v_ptr: Any,
        do_ptr: Any,
        dq_ptr: Any,
        dk_ptr: Any,
        dv_ptr: Any,
        lse_ptr: Any,
        delta_ptr: Any,
        sm_scale: Any,
        stride_b: Any,
        stride_h: Any,
        stride_s: Any,
        stride_d: Any,
        n_ctx: Any,
        q_heads: Any,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
    ) -> None:
        start_n = tl.program_id(0)
        off_bh = tl.program_id(1)
        off_b = off_bh // q_heads
        off_h = off_bh % q_heads
        base = off_b * stride_b + off_h * stride_h

        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        n_mask = offs_n < n_ctx

        kv_ptrs = base + offs_n[:, None] * stride_s + offs_d[None, :] * stride_d
        k = tl.load(k_ptr + kv_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptr + kv_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)

        lo = start_n * BLOCK_N if CAUSAL else 0
        for start_m in range(lo, n_ctx, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            m_mask = offs_m < n_ctx
            q_ptrs = base + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d
            q = tl.load(q_ptr + q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
            do = tl.load(do_ptr + q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

            qk = tl.dot(q, tl.trans(k), allow_tf32=False) * sm_scale
            qk += tl.where(n_mask[None, :], 0.0, float("-inf"))
            if CAUSAL:
                qk += tl.where(offs_m[:, None] >= offs_n[None, :], 0.0, float("-inf"))
            lse_m = tl.load(lse_ptr + off_bh * n_ctx + offs_m, mask=m_mask, other=0.0)
            p = tl.exp(qk - lse_m[:, None])

            dv += tl.dot(tl.trans(p), do, allow_tf32=False)
            dp = tl.dot(do, tl.trans(v), allow_tf32=False)
            delta_m = tl.load(delta_ptr + off_bh * n_ctx + offs_m, mask=m_mask, other=0.0)
            ds = p * (dp - delta_m[:, None]) * sm_scale
            dk += tl.dot(tl.trans(ds), q, allow_tf32=False)
            dq = tl.dot(ds, k, allow_tf32=False)
            tl.atomic_add(dq_ptr + q_ptrs, dq, mask=m_mask[:, None])

        tl.store(dk_ptr + kv_ptrs, dk.to(dk_ptr.dtype.element_ty), mask=n_mask[:, None])
        tl.store(dv_ptr + kv_ptrs, dv.to(dv_ptr.dtype.element_ty), mask=n_mask[:, None])
else:  # pragma: no cover - triton not installed on the dev box
    _preprocess = None
    kernel = None


def roofline_spec(shape: dict[str, Any], dtype: str) -> dict[str, Any] | None:
    """The FLOP / HBM-byte roofline spec for ``shape={"B","H","S","D"}`` (plus
    an optional ``"causal"`` bool) at ``dtype``, or ``None`` if a dimension is
    missing. Pure; no GPU needed."""
    return roofline_spec_for(KERNEL_KEY, shape, dtype)


def _resolve(config: dict[str, int] | None) -> dict[str, int]:
    return {**CONFIG, **(config or {})}


def _expanded_qkv(
    b: int, h: int, s: int, d: int, h_kv: int, torch_dtype: Any
) -> tuple[Any, Any, Any]:
    """Random Q ``(b, h, s, d)`` and K/V expanded to ``h`` heads (GQA)."""
    import torch

    q = torch.randn((b, h, s, d), device="cuda", dtype=torch_dtype)
    k = torch.randn((b, h_kv, s, d), device="cuda", dtype=torch_dtype)
    v = torch.randn((b, h_kv, s, d), device="cuda", dtype=torch_dtype)
    group = h // h_kv
    return q, k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)


def _reference_fwd(q: Any, k: Any, v: Any, causal: bool) -> tuple[Any, Any]:
    """``O`` (in the input dtype) and its fp32 log-sum-exp for the given
    (head-expanded) Q/K/V -- the backward kernel's inputs, formed untimed and
    consistently in fp32 so ``O`` and ``LSE`` correspond exactly."""
    import torch

    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = (q.float() @ k.float().transpose(-1, -2)) * scale
    if causal:
        s = q.shape[-2]
        cmask = torch.triu(torch.ones(s, s, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(cmask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # (b, h, s), fp32
    p = torch.softmax(scores, dim=-1)
    o = (p @ v.float()).to(q.dtype)
    return o, lse


def _alloc_bufs(q: Any) -> dict[str, Any]:
    import torch

    b, h, s, d = q.shape
    return {
        "delta": torch.empty((b * h, s), device="cuda", dtype=torch.float32),
        "dq": torch.zeros((b, h, s, d), device="cuda", dtype=torch.float32),
        "dk": torch.empty_like(q),
        "dv": torch.empty_like(q),
    }


def _launch_bwd(
    q: Any,
    k: Any,
    v: Any,
    o: Any,
    lse: Any,
    do: Any,
    causal: bool,
    cfg: dict[str, int],
    bufs: dict[str, Any],
) -> tuple[Any, Any, Any]:
    b, h, s, d = q.shape
    delta, dq, dk, dv = bufs["delta"], bufs["dq"], bufs["dk"], bufs["dv"]
    dq.zero_()  # atomics accumulate into it
    sm_scale = 1.0 / (d**0.5)
    block_m, block_n = cfg["BLOCK_M"], cfg["BLOCK_N"]
    assert _preprocess is not None and kernel is not None
    _preprocess[(triton.cdiv(s, block_m), b * h)](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        s,
        h,
        BLOCK_M=block_m,
        BLOCK_D=d,
    )
    kernel[(triton.cdiv(s, block_n), b * h)](
        q,
        k,
        v,
        do,
        dq,
        dk,
        dv,
        lse.reshape(b * h, s),
        delta,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        s,
        h,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=d,
        CAUSAL=causal,
    )
    return dq.to(q.dtype), dk, dv


def _group_reduce(grad: Any, h_kv: int) -> Any:
    """Sum a head-expanded ``dK`` / ``dV`` back to ``h_kv`` heads (grouped-query
    attention: each K/V head's gradient is the sum over its query group)."""
    b, h, s, d = grad.shape
    return grad.view(b, h_kv, h // h_kv, s, d).sum(dim=2)


def check_numerics(cell: dict[str, Any], config: dict[str, int] | None = None) -> dict[str, Any]:
    """Run the Triton backward and torch autograd once (untimed) and compare
    ``dQ`` / ``dK`` / ``dV`` (reduced to ``h_kv`` heads). Returns
    ``{"max_abs_err", "max_rel_err", "allclose"}``. Needs Triton, PyTorch, and
    a CUDA device."""
    require_live_deps("attention_bwd")
    import torch
    import torch.nn.functional as F

    b, h, s, d, h_kv, causal = attention_dims(cell)
    torch_dtype = attention_torch_dtype(cell.get("dtype", "bf16"))
    q, k, v = _expanded_qkv(b, h, s, d, h_kv, torch_dtype)
    do = torch.randn_like(q)

    o, lse = _reference_fwd(q, k, v, causal)
    dq_t, dk_t, dv_t = _launch_bwd(q, k, v, o, lse, do, causal, _resolve(config), _alloc_bufs(q))

    qa, ka, va = (x.clone().detach().requires_grad_(True) for x in (q, k, v))
    F.scaled_dot_product_attention(qa, ka, va, is_causal=causal).backward(do)
    grads_t = (dq_t, _group_reduce(dk_t, h_kv), _group_reduce(dv_t, h_kv))
    grads_a = (qa.grad, _group_reduce(ka.grad, h_kv), _group_reduce(va.grad, h_kv))

    tol = 3e-3 if torch_dtype == torch.float32 else 3e-2
    max_abs = 0.0
    max_rel = 0.0
    ok = True
    for gt, ga in zip(grads_t, grads_a, strict=True):
        gt32, ga32 = gt.to(torch.float32), ga.to(torch.float32)
        err = (gt32 - ga32).abs()
        max_abs = max(max_abs, float(err.max()))
        max_rel = max(max_rel, float((err / ga32.abs().clamp_min(1e-6)).max()))
        ok = ok and torch.allclose(gt32, ga32, atol=tol, rtol=tol)
    return {"max_abs_err": max_abs, "max_rel_err": max_rel, "allclose": ok}


def run(cell: dict[str, Any], config: dict[str, int] | None = None) -> Result:
    """Time this attention backward at ``cell``'s shape/dtype (``cell`` may also
    carry ``causal`` / ``h_kv``). Both sides are backward-only: the Triton path
    from given ``Q,K,V,O,LSE,dO``; the baseline as ``autograd.grad`` over one
    untimed SDPA forward. Needs Triton, PyTorch, and a CUDA device."""
    require_live_deps("attention_bwd")
    import torch
    import torch.nn.functional as F

    b, h, s, d, h_kv, causal = attention_dims(cell)
    dtype_name = cell.get("dtype", "bf16")
    torch_dtype = attention_torch_dtype(dtype_name)
    cfg = _resolve(config)
    q, k, v = _expanded_qkv(b, h, s, d, h_kv, torch_dtype)
    do = torch.randn_like(q)
    o, lse = _reference_fwd(q, k, v, causal)
    bufs = _alloc_bufs(q)

    samples_us = [
        t * 1000.0 for t in time_kernel(lambda: _launch_bwd(q, k, v, o, lse, do, causal, cfg, bufs))
    ]

    qa, ka, va = (x.clone().detach().requires_grad_(True) for x in (q, k, v))
    o_sdpa = F.scaled_dot_product_attention(qa, ka, va, is_causal=causal)  # one forward, untimed
    baseline_us = [
        t * 1000.0
        for t in time_kernel(
            lambda: torch.autograd.grad(
                (o_sdpa,), (qa, ka, va), grad_outputs=(do,), retain_graph=True
            )
        )
    ]
    triton_p50 = sorted(samples_us)[len(samples_us) // 2]
    baseline_p50 = sorted(baseline_us)[len(baseline_us) // 2]

    spec = roofline_spec({"B": b, "H": h, "S": s, "D": d, "causal": causal}, dtype_name)

    return assemble_result(
        kernel_name="attention_bwd",
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
        baseline="sdpa_backward",
        baseline_pct=(baseline_p50 / triton_p50) if triton_p50 > 0 else None,
    )
