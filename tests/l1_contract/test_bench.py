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
    d = r.to_dict()
    assert 198.0 < d["timing"]["p50_us"] < 202.0
    assert d["timing"]["p10_us"] <= d["timing"]["p50_us"] <= d["timing"]["p90_us"]
    assert d["timing"]["invalidated_samples"] == 0
    assert d["timing"]["n_warmup_to_steady"] == 0
    assert d["flags"] == []
    assert d["clocks"]["locked"] is True
    assert d["machine"]["sm_arch"] == "sm_89"
    assert r.validate() == []


def test_unlocked_throttled_recording_is_flagged_and_cleaned() -> None:
    d = bench(recording=_rec("unlocked_throttled.jsonl"), batches=40).to_dict()
    assert d["timing"]["invalidated_samples"] == 2
    assert d["timing"]["n_samples"] == 38
    assert 198.0 < d["timing"]["p50_us"] < 202.0
    assert d["throttle_reasons"] == ["SW_POWER_CAP"]
    assert "clocks-unlocked" in d["flags"]
    assert "throttled-samples-dropped" in d["flags"]


def test_cold_ramp_recording_is_trimmed() -> None:
    d = bench(recording=_rec("cold_ramp.jsonl"), batches=70).to_dict()
    assert d["timing"]["n_warmup_to_steady"] > 0
    assert d["timing"]["n_samples"] < 70
    assert 198.0 < d["timing"]["p50_us"] < 203.0


def test_bench_without_a_recording_is_not_implemented_yet() -> None:
    with pytest.raises(NotImplementedError):
        bench(lambda: None)


def test_bench_rejects_a_malformed_recording() -> None:
    with pytest.raises(ValueError):
        bench(recording="{not json}", batches=1)
