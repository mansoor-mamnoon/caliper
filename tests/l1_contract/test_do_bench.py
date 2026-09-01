"""L1: the Triton-compatible ``caliper.do_bench`` shim over a recording."""

from __future__ import annotations

from pathlib import Path

import pytest

from caliper import bench, do_bench

pytestmark = pytest.mark.l1

FIXTURES = Path(__file__).resolve().parents[2] / "crates" / "caliper-gpu" / "fixtures" / "bench"


def _rec(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_return_modes_come_back_in_milliseconds() -> None:
    r = bench(recording=_rec("happy.jsonl"), batches=40)
    p50_ms = r.p50_us / 1000.0  # type: ignore[operator]

    assert do_bench(recording=_rec("happy.jsonl"), batches=40) == pytest.approx(
        r.mean_us / 1000.0  # type: ignore[operator]
    )
    assert do_bench(
        recording=_rec("happy.jsonl"), batches=40, return_mode="median"
    ) == pytest.approx(p50_ms)
    lo = do_bench(recording=_rec("happy.jsonl"), batches=40, return_mode="min")
    hi = do_bench(recording=_rec("happy.jsonl"), batches=40, return_mode="max")
    assert isinstance(lo, float) and isinstance(hi, float)
    assert lo <= p50_ms <= hi
    assert 0.15 < p50_ms < 0.25  # ~200 us


def test_quantiles_override_return_mode_and_return_a_list() -> None:
    out = do_bench(recording=_rec("happy.jsonl"), batches=40, quantiles=[0.5, 0.2, 0.8])
    assert isinstance(out, list)
    assert len(out) == 3
    assert out[1] <= out[0] <= out[2]  # p20 <= p50 <= p80
    assert all(0.15 < v < 0.25 for v in out)


def test_triton_compat_kwargs_are_accepted() -> None:
    # An unmodified Triton call site passes warmup/rep/grad_to_none/fast_flush.
    val = do_bench(
        lambda: None,
        warmup=25,
        rep=100,
        grad_to_none=[1, 2, 3],
        fast_flush=True,
        quantiles=[0.5],
        recording=_rec("happy.jsonl"),
        batches=40,
    )
    assert isinstance(val, list) and len(val) == 1


def test_a_bad_return_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="return_mode"):
        do_bench(recording=_rec("happy.jsonl"), return_mode="p99")


def test_a_live_callable_needs_torch_and_cuda() -> None:
    # No CUDA on the dev box: the live path must degrade to a clear error, not
    # crash. (The live timing loop itself is exercised on a CUDA host.)
    with pytest.raises(NotImplementedError, match="CUDA"):
        do_bench(lambda: None)
