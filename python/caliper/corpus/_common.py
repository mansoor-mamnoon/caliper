"""Shared plumbing for the reference kernel corpus (``corpus/kernels/*.py``).

Every corpus kernel is measured through :func:`caliper.live_timing_ms` -- the
same CUDA-event loop ``do_bench`` uses -- not through ``bench()``'s device-layer
ports (those drive the CUDA-C++ oracle kernels O1-O7 and their real launcher is
still a stub). So a corpus kernel runs without waiting on the rest of the
on-device work: it needs only PyTorch, Triton, and a CUDA device. On-device
verification happens on Colab (``docs/plan.md`` s0.5).

Because the run bypasses the Rust reduction pipeline (no clock lock, no L2
flush accounting, no throttle detection, no steady-state trim), every record
this module assembles is flagged ``corpus-live-timing`` so a reader knows which
guarantees it does *not* carry.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

from caliper import Result, _core
from caliper.api import live_timing_ms

__all__ = [
    "TRITON_PIN",
    "AttentionDims",
    "TritonPin",
    "assemble_result",
    "attention_dims",
    "attention_torch_dtype",
    "content_hash",
    "dim",
    "has_module",
    "require_live_deps",
    "roofline_spec_for",
    "time_kernel",
    "torch_machine",
]

#: dtypes the attention corpus kernels (and their SDPA baselines) accept.
ATTENTION_DTYPES = ("bf16", "fp16", "fp32")


class TritonPin(NamedTuple):
    """The Triton commit a kernel's API usage was written and last checked
    against -- not a claim that the kernel body is copied from that commit."""

    repo: str
    commit: str
    date: str  # ISO 8601 UTC, when `commit` was fetched


# https://api.github.com/repos/triton-lang/triton/commits/main, fetched once at
# the time this corpus was written. Triton's kernel-authoring API is what this
# pin tracks; each kernel here is caliper's own implementation.
TRITON_PIN = TritonPin(
    repo="triton-lang/triton",
    commit="58895270e6230491acc15fe7ba6d4c849e838db1",
    date="2026-09-01T01:45:09Z",
)


def content_hash(source_file: str | Path) -> str:
    """``sha256:<hex>`` of a kernel module's own source file, for
    ``kernel.source_hash`` and the autotune cache key."""
    data = Path(source_file).read_bytes()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def has_module(name: str) -> bool:
    """Whether ``name`` is importable, without actually importing it."""
    return importlib.util.find_spec(name) is not None


def dim(shape: dict[str, Any], *names: str) -> int:
    """The first of ``names`` present in ``shape``, as an ``int``. Lets a
    kernel accept a dimension under either case (``"m"`` from a ``sweep`` cell,
    ``"M"`` from a direct call)."""
    for name in names:
        if name in shape:
            return int(shape[name])
    raise KeyError(f"shape is missing one of {names}: {shape!r}")


def roofline_spec_for(kernel_key: str, shape: dict[str, Any], dtype: str) -> dict[str, Any] | None:
    """The FLOP / HBM-byte roofline spec for ``kernel_key`` at ``shape`` and
    ``dtype`` (delegating to ``caliper._core``), or ``None`` if a dimension is
    missing. Pure; no GPU needed."""
    spec_json = _core.corpus_roofline_spec(kernel_key, json.dumps(shape), dtype)
    return dict(json.loads(spec_json)) if spec_json is not None else None


class AttentionDims(NamedTuple):
    """The problem an attention corpus kernel is being asked to run."""

    b: int
    h: int
    s: int
    d: int
    h_kv: int  # K/V heads (< h for grouped-query attention); defaults to h
    causal: bool


def _cell_get(cell: dict[str, Any], key: str, default: Any) -> Any:
    """``cell[key]`` or ``cell["shape"][key]`` (a cell may carry attention
    knobs at either level), falling back to ``default`` only when neither is
    set -- an explicit ``False`` / ``0`` is honoured."""
    for src in (cell, cell.get("shape", {})):
        if key in src:
            return src[key]
    return default


def attention_dims(cell: dict[str, Any]) -> AttentionDims:
    """``(b, h, s, d, h_kv, causal)`` from a cell. Dimensions come from
    ``cell["shape"]`` under either case (``"s"`` from a ``sweep`` cell, ``"S"``
    from a direct call, via :func:`dim`); ``h_kv`` / ``causal`` come from the
    cell or its shape and default to ``h`` (plain multi-head) / ``False``."""
    shape = cell["shape"]
    b, h, s, d = (
        dim(shape, "b", "B"),
        dim(shape, "h", "H"),
        dim(shape, "s", "S"),
        dim(shape, "d", "D"),
    )
    h_kv = int(_cell_get(cell, "h_kv", h))
    if h_kv <= 0 or h % h_kv != 0:
        raise ValueError(f"h={h} must be a positive multiple of h_kv={h_kv}")
    return AttentionDims(b, h, s, d, h_kv, bool(_cell_get(cell, "causal", False)))


def attention_torch_dtype(dtype_name: str) -> Any:
    """The ``torch`` dtype for an attention corpus kernel. Raises
    ``NotImplementedError`` for anything outside :data:`ATTENTION_DTYPES` (the
    fp8 path for L4 is a follow-up), rather than silently downcasting -- and it
    does so before importing torch, so the rejection is testable off-GPU."""
    if dtype_name not in ATTENTION_DTYPES:
        raise NotImplementedError(
            f"the attention corpus kernels support {ATTENTION_DTYPES}; "
            f"{dtype_name!r} (e.g. the fp8 path for L4) is a follow-up."
        )
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]


def require_live_deps(kernel_name: str) -> None:
    """Raise ``NotImplementedError`` unless PyTorch, Triton, and a CUDA device
    are all available. The PyTorch/CUDA half delegates to
    :func:`caliper.api._check_live_deps` so every live-callable timing path
    (``do_bench``, the corpus kernels) fails the same way."""
    from caliper.api import _check_live_deps

    caller = f"caliper.corpus.kernels.{kernel_name}.run()"
    _check_live_deps(caller)
    if not has_module("triton"):
        raise NotImplementedError(f"{caller} needs Triton (pip install triton).")


def torch_machine() -> dict[str, Any]:
    """A best-effort machine fingerprint built from PyTorch device
    introspection alone -- no NVML, so it works whether or not the Rust
    device-layer ports are wired up on this host."""
    import torch

    from caliper.api import toolchain

    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    tc = toolchain()
    return {
        "gpu_name": props.name,
        "sm_arch": f"sm_{major}{minor}",
        "vram_mib": props.total_memory // (1024 * 1024),
        "sm_count": props.multi_processor_count,
        "cuda_runtime": torch.version.cuda,
        "toolkit": {
            "triton": tc["triton"],
            "torch": tc["torch"],
            "ptxas": tc["ptxas"],
            "nvcc": tc["nvcc"],
        },
    }


def assemble_result(
    *,
    kernel_name: str,
    kernel_impl: str,
    dtype: str,
    layout: str | None,
    shape: dict[str, Any],
    source_hash: str,
    autotune_config: dict[str, Any],
    samples_us: list[float],
    machine: dict[str, Any],
    flops: float | None = None,
    bytes_hbm: float | None = None,
    baseline: str | None = None,
    baseline_pct: float | None = None,
) -> Result:
    """Build a :class:`~caliper.Result` from raw per-launch timing samples
    (microseconds), the kernel identity, the ``machine`` fingerprint (see
    :func:`torch_machine`), and (optionally) the FLOP / HBM-byte counts needed
    for a roofline. Pure -- no GPU -- so an off-device test can drive it.
    """
    summary = _core.summarize(samples_us)  # {n, min, p10, p50, p90, max, mean, mad, [cov]}

    record: dict[str, Any] = Result.default().to_dict()
    record["measured_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record["kernel"] = {
        "name": kernel_name,
        "impl": kernel_impl,
        "source_hash": source_hash,
        "autotune_config": autotune_config,
        "dtype": dtype,
        "shape": shape,
        "layout": layout,
    }
    record["timing"] = {
        "p10_us": summary["p10"],
        "p50_us": summary["p50"],
        "p90_us": summary["p90"],
        "mad_us": summary["mad"],
        "mean_us": summary["mean"],
        "min_us": summary["min"],
        "max_us": summary["max"],
        "n_samples": int(summary["n"]),
        "wall_p50_us": None,
        "launch_overhead_us": None,
        "n_warmup_to_steady": None,
        "invalidated_samples": None,
        # `summary["cov"]` is intra-run (single-pass) dispersion, not the
        # cross-pass quantity this field names; this path only ever runs one
        # pass, so cross_pass_cov genuinely doesn't apply here.
        "cross_pass_cov": None,
    }
    record["machine"] = machine
    record["flags"] = ["clocks-unlocked", "corpus-live-timing"]

    if flops is not None and bytes_hbm is not None:
        seconds = summary["p50"] * 1e-6
        analysis = json.loads(
            _core.roofline_analyze(machine["sm_arch"], dtype, flops, bytes_hbm, seconds)
        )
        pct = analysis["roofline_pct"]
        record["roofline"] = {
            "achieved_tflops": analysis["achieved_tflops"],
            # match the Rust reduce pipeline's clamp (roofline.rs: 0.0 ..= 1.5).
            "roofline_pct": (max(0.0, min(pct, 1.5)) if isinstance(pct, (int, float)) else None),
            "achieved_gbps": analysis["achieved_gbps"],
            "arithmetic_intensity": analysis["arithmetic_intensity"],
            "ridge_point": analysis["ridge_point"],
            "bound": analysis["bound"],
            "baseline_pct": baseline_pct,
            "baseline": baseline,
        }

    return Result.from_dict(record)


# re-exported so a kernel module's `run()` needs only `caliper.corpus._common`
time_kernel = live_timing_ms
