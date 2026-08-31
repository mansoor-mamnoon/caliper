"""L0: the Rust roofline model, through the bindings.

Exhaustive coverage is in `cargo test`; these check the binding surface, that
the peaks table answers for every supported architecture, and that the
regime classification matches a couple of hand-computed cases.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from caliper import _core

pytestmark = pytest.mark.l0

ARCHES = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_120", "cdna3"]


def _analyze(
    arch: str, dtype: str, flops: float, bytes_hbm: float, seconds: float
) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(
        _core.roofline_analyze(arch, dtype, flops, bytes_hbm, seconds)
    )
    return result


@pytest.mark.parametrize("arch", ARCHES)
def test_every_arch_has_a_bandwidth_and_fp32_peak(arch: str) -> None:
    assert _core.peak_hbm_gbps(arch) is not None
    assert _core.peak_compute_tflops(arch, "fp32") is not None


def test_ridge_point_is_peak_compute_over_bandwidth() -> None:
    # A100 bf16: 312 TFLOP/s / 2039 GB/s.
    r = _analyze("sm_80", "bf16", 1e12, 1e9, 1e-3)
    assert r["ridge_point"] == pytest.approx(312e12 / 2039e9, rel=1e-6)


def test_dense_matmul_past_the_ridge_is_compute_bound() -> None:
    n = 4096.0
    flops = 2.0 * n * n * n
    bytes_hbm = 3.0 * n * n * 2.0  # bf16 I/O
    r = _analyze("sm_80", "bf16", flops, bytes_hbm, 0.50e-3)
    assert r["bound"] == "compute"
    assert r["arithmetic_intensity"] > r["ridge_point"]
    assert 0.80 < r["roofline_pct"] < 1.0


def test_streaming_triad_below_the_ridge_is_memory_bound() -> None:
    bytes_hbm = 3.0 * 1024.0 * 1024.0 * 1024.0
    flops = bytes_hbm / 12.0
    r = _analyze("sm_80", "fp32", flops, bytes_hbm, 1.60e-3)
    assert r["bound"] == "memory"
    assert r["arithmetic_intensity"] < r["ridge_point"]
    assert r["roofline_pct"] > 0.80


def test_far_below_both_ceilings_is_latency_bound() -> None:
    r = _analyze("sm_90", "fp16", 1e9, 1e6, 1e-3)
    assert r["bound"] == "latency"


def test_unknown_arch_or_dtype_is_unknown_but_keeps_achieved() -> None:
    r = _analyze("sm_42", "fp16", 1e12, 1e9, 1e-3)
    assert r["bound"] == "unknown"
    assert r["achieved_tflops"] > 0.0
    assert r["roofline_pct"] is None
    # Volta has no FP8 tensor path.
    assert _analyze("sm_70", "fp8", 1e12, 1e9, 1e-3)["bound"] == "unknown"


def test_achieved_bandwidth_matches_the_o2_formula() -> None:
    bytes_per_array = 1024.0 * 1024.0 * 1024.0
    p50_us = 3300.0
    r = _analyze("sm_89", "fp32", 1.0, 3.0 * bytes_per_array, p50_us * 1e-6)
    assert r["achieved_gbps"] == pytest.approx(3.0 * bytes_per_array / (p50_us * 1e3), rel=1e-9)
