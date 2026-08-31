"""L0: the Rust result schema, exercised through the Python bindings."""

from __future__ import annotations

import json

import pytest

from caliper import Result, _core, schema_version

pytestmark = pytest.mark.l0


def test_schema_version_is_one() -> None:
    assert schema_version() == "1"
    assert _core.schema_version() == "1"


def test_default_record_has_versions_and_empty_sections() -> None:
    r = Result.default()
    assert r["schema_version"] == "1"
    assert r["caliper_version"]  # non-empty
    assert r["throttle_reasons"] == []
    assert r["timing"]["p50_us"] is None
    assert r.validate() == []


def test_default_round_trips_through_dict() -> None:
    r = Result.default()
    assert Result.from_dict(r.to_dict()) == r


def test_populated_record_round_trips_through_json() -> None:
    d = Result.default().to_dict()
    d["measured_at"] = "2026-01-02T03:04:05Z"
    d["kernel"] |= {
        "name": "matmul_kernel",
        "impl": "triton",
        "autotune_config": {"BLOCK_M": 128, "num_warps": 8},
        "shape": {"M": 4096, "N": 4096, "K": 4096},
        "dtype": "bf16",
    }
    d["timing"] |= {"p10_us": 241.0, "p50_us": 243.2, "p90_us": 250.1, "n_samples": 300}
    d["roofline"] |= {"achieved_tflops": 565.0, "roofline_pct": 0.86, "bound": "compute"}
    d["ptxas"]["regs_per_thread"] = 168
    d["throttle_reasons"] = ["SW_POWER_CAP"]

    r = Result.from_dict(d)
    assert Result.from_json(r.to_json()) == r
    assert r.to_json() == _core.normalize_record_json(r.to_json())  # canonical is stable
    assert r.validate() == []


def test_unknown_keys_are_dropped_and_missing_sections_filled() -> None:
    r = Result.from_dict({"kernel": {"impl": "cuda", "bogus": 1}, "surprise": True})
    assert r["kernel"]["impl"] == "cuda"
    assert "bogus" not in r["kernel"]
    assert "surprise" not in r.to_dict()
    assert r["timing"]["p50_us"] is None


def test_missing_version_is_filled_from_current() -> None:
    r = Result.from_dict({"kernel": {"impl": "cuda"}})
    assert r["schema_version"] == "1"


def test_validate_reports_each_class_of_problem() -> None:
    d = Result.default().to_dict()
    d["schema_version"] = "99"
    d["timing"]["p10_us"] = 9.0
    d["timing"]["p50_us"] = 2.0
    d["roofline"]["roofline_pct"] = 4.0
    d["occupancy"]["achieved"] = -0.1
    problems = _core.validate_record_json(json.dumps(d))
    assert len(problems) == 4, problems


def test_invalid_json_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _core.normalize_record_json("{not valid json")
