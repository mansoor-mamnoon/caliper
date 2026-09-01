"""L0: the selftest report model, through the bindings.

Exhaustive coverage is in `cargo test`; these check the binding surface and the
Appendix-E report shape.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from caliper import _core

pytestmark = pytest.mark.l0


def _check(name: str, status: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": "x", **extra}


def _assemble(checks: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(_core.selftest_assemble("{}", json.dumps(checks)))
    return report


def test_all_pass_is_a_pass_reduced_coverage() -> None:
    r = _assemble([_check("o1_duration_linearity", "PASS"), _check("o3_fma_peak", "PASS")])
    assert r["result"] == "PASS"
    assert r["coverage"] == "reduced"
    assert r["not_validated"] == []
    assert r["schema_version"] == _core.schema_version()
    assert _core.validate_selftest_json(json.dumps(r)) == []


def test_a_failing_check_downgrades_to_fail() -> None:
    r = _assemble([_check("o1_duration_linearity", "PASS"), _check("o3_fma_peak", "FAIL")])
    assert r["result"] == "FAIL"


def test_an_error_beats_a_fail_and_all_skips_is_an_error() -> None:
    assert _assemble([_check("a", "FAIL"), _check("b", "ERROR")])["result"] == "ERROR"
    both_skip = _assemble([_check("o1", "SKIP"), _check("o2", "SKIP")])
    assert both_skip["result"] == "ERROR"
    assert set(both_skip["not_validated"]) == {"o1", "o2"}


def test_nsys_pass_lifts_coverage_to_full() -> None:
    r = _assemble([_check("o1_duration_linearity", "PASS"), _check("vs_nsys", "PASS")])
    assert r["coverage"] == "full"


def test_validate_catches_an_inconsistent_report() -> None:
    r = _assemble([_check("o1_duration_linearity", "PASS")])
    r["result"] = "FAIL"
    assert _core.validate_selftest_json(json.dumps(r)) != []


@pytest.mark.parametrize("bad", ["{not json", "null", '{"name": 1}'])
def test_selftest_assemble_rejects_bad_checks(bad: str) -> None:
    with pytest.raises(ValueError):
        _core.selftest_assemble("{}", bad)


def test_selftest_assemble_rejects_a_bad_machine() -> None:
    with pytest.raises(ValueError):
        _core.selftest_assemble("42", "[]")
