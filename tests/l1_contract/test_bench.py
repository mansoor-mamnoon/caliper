"""L1: bench() end to end over a recorded device session, through the bindings."""

from __future__ import annotations

from pathlib import Path

import pytest

from caliper import Result, bench

pytestmark = pytest.mark.l1

FIXTURES = Path(__file__).resolve().parents[2] / "crates" / "caliper-gpu" / "fixtures" / "bench"


def _rec(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_happy_recording_yields_a_clean_locked_result() -> None:
    r = bench(recording=_rec("happy.jsonl"), batches=40)
    assert isinstance(r, Result)
    # read accessors (the frozen public surface)
    assert 198.0 < r.p50_us < 202.0  # type: ignore[operator]
    assert r.p10_us <= r.p50_us <= r.p90_us  # type: ignore[operator]
    assert r.timing["invalidated_samples"] == 0
    assert r.timing["n_warmup_to_steady"] == 0
    assert r.flags == []
    assert r.clocks["locked"] is True
    assert r.machine["sm_arch"] == "sm_89"
    assert r.ptxas["regs_per_thread"] == 168  # from the module probe
    assert r.ptxas.spill_stores_bytes == 0
    assert r.schema_version == "1"
    assert r.validate() == []


def test_ptxas_unavailable_recording_is_flagged_not_failed() -> None:
    r = bench(recording=_rec("ptxas_unavailable.jsonl"), batches=40)
    assert "ptxas-unavailable" in r.flags
    assert r.ptxas["regs_per_thread"] is None
    assert 198.0 < r.p50_us < 202.0  # type: ignore[operator]


def test_unlocked_throttled_recording_is_flagged_and_cleaned() -> None:
    r = bench(recording=_rec("unlocked_throttled.jsonl"), batches=40)
    assert r.timing["invalidated_samples"] == 2
    assert r.timing["n_samples"] == 38
    assert 198.0 < r.p50_us < 202.0  # type: ignore[operator]
    assert r.throttle_reasons == ["SW_POWER_CAP"]
    assert "clocks-unlocked" in r.flags
    assert "throttled-samples-dropped" in r.flags


def test_cold_ramp_recording_is_trimmed() -> None:
    r = bench(recording=_rec("cold_ramp.jsonl"), batches=70)
    assert r.timing["n_warmup_to_steady"] > 0
    assert r.timing["n_samples"] < 70
    assert 198.0 < r.p50_us < 203.0  # type: ignore[operator]


def test_fixed_warmup_trims_exactly_n() -> None:
    r = bench(recording=_rec("happy.jsonl"), batches=40, warmup=12)
    assert r.timing["n_warmup_to_steady"] == 12
    assert r.timing["n_samples"] == 28


def test_a_hard_lock_error_degrades_instead_of_raising() -> None:
    r = bench(recording=_rec("lock_error.jsonl"), batches=40)
    assert "clocks-unlocked" in r.flags
    assert r.clocks["locked"] is False


def test_cuda_graph_mode_is_accepted() -> None:
    assert bench(recording=_rec("happy.jsonl"), batches=40, cuda_graph="off").flags == []
    with pytest.raises(ValueError):
        bench(recording=_rec("happy.jsonl"), batches=40, cuda_graph="sometimes")


def test_corpus_target_sets_the_kernel_key() -> None:
    r = bench("corpus:o1", recording=_rec("oracle_o1.jsonl"), batches=40)
    assert r.kernel["name"] == "oracle:busy"


def test_unknown_corpus_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown corpus target"):
        bench("corpus:o9", recording=_rec("oracle_o1.jsonl"), batches=40)


def test_bench_without_a_recording_is_not_implemented_yet() -> None:
    with pytest.raises(NotImplementedError):
        bench(lambda: None)


@pytest.mark.parametrize("bad_warmup", ["sometimes", -3, 1.5])
def test_bench_rejects_a_bad_warmup(bad_warmup: object) -> None:
    with pytest.raises(ValueError):
        bench(recording=_rec("happy.jsonl"), batches=40, warmup=bad_warmup)  # type: ignore[arg-type]


def test_bench_rejects_a_malformed_recording() -> None:
    with pytest.raises(ValueError):
        bench(recording="{not json}", batches=1)


def test_bench_rejects_a_recording_with_leftover_calls() -> None:
    with pytest.raises(ValueError):
        bench(recording=_rec("trailing_call.jsonl"), batches=40)
