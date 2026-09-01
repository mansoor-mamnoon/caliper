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

APPENDIX_E_KEYS = {
    "schema_version",
    "caliper_version",
    "machine",
    "result",
    "coverage",
    "checks",
    "not_validated",
}


def _check(name: str, status: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": "x", **extra}


def _assemble(
    checks: list[dict[str, Any]], not_validated: list[str] | None = None
) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(
        _core.selftest_assemble("{}", json.dumps(checks), json.dumps(not_validated or []))
    )
    return report


def test_all_pass_is_a_pass_and_matches_appendix_e() -> None:
    r = _assemble([_check("o1_duration_linearity", "PASS"), _check("o3_fma_peak", "PASS")])
    assert r["result"] == "PASS"
    assert r["coverage"] == "reduced"
    assert set(r) == APPENDIX_E_KEYS  # no stray keys (e.g. exit_code)
    assert r["schema_version"] == _core.schema_version()
    assert _core.validate_selftest_json(json.dumps(r)) == []


def test_a_failing_scored_check_downgrades_to_fail() -> None:
    assert (
        _assemble([_check("o1_duration_linearity", "PASS"), _check("o3_fma_peak", "FAIL")])[
            "result"
        ]
        == "FAIL"
    )


def test_a_context_line_is_not_a_scored_pass() -> None:
    r = _assemble([_check("device_present", "PASS"), _check("o1_duration_linearity", "SKIP")])
    assert r["result"] == "ERROR"  # nothing in the suite actually ran


def test_all_skips_is_an_error_not_a_pass() -> None:
    assert (
        _assemble([_check("o1_duration_linearity", "SKIP"), _check("o3_fma_peak", "SKIP")])[
            "result"
        ]
        == "ERROR"
    )


def test_nsys_pass_lifts_coverage_to_full() -> None:
    assert (
        _assemble([_check("o1_duration_linearity", "PASS"), _check("vs_nsys", "PASS")])["coverage"]
        == "full"
    )


def test_not_validated_only_accepts_capability_tokens() -> None:
    ok = _assemble([_check("o1_duration_linearity", "PASS")], ["clock_lock", "ncu_crosscheck"])
    assert _core.validate_selftest_json(json.dumps(ok)) == []

    bad = _assemble([_check("o1_duration_linearity", "PASS")], ["o2_bandwidth"])
    assert _core.validate_selftest_json(json.dumps(bad)) != []


def test_validate_catches_an_inconsistent_report() -> None:
    r = _assemble([_check("o1_duration_linearity", "PASS")])
    r["result"] = "FAIL"
    assert _core.validate_selftest_json(json.dumps(r)) != []


def test_validate_rejects_a_fabricated_pass() -> None:
    r = _assemble([_check("o1_duration_linearity", "SKIP")])
    assert r["result"] == "ERROR"
    r["result"] = "PASS"
    problems = _core.validate_selftest_json(json.dumps(r))
    assert any("no scored check passed" in p for p in problems)


@pytest.mark.parametrize("bad", ["{not json", "null", '{"name": 1}'])
def test_selftest_assemble_rejects_bad_checks(bad: str) -> None:
    with pytest.raises(ValueError):
        _core.selftest_assemble("{}", bad, "[]")


def test_selftest_assemble_rejects_a_bad_machine() -> None:
    with pytest.raises(ValueError):
        _core.selftest_assemble("42", "[]", "[]")
