"""L0: the machine-fingerprint completeness check and toolchain parsing.

Exhaustive coverage is in `cargo test`; these check the binding surface and the
Python toolchain probe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from caliper import _core, _toolchain, api

pytestmark = pytest.mark.l0

BENCH = Path(__file__).resolve().parents[2] / "crates" / "caliper-gpu" / "fixtures" / "bench"


def _full_machine() -> dict[str, Any]:
    raw = _core.fingerprint_replay((BENCH / "happy.jsonl").read_text())
    machine: dict[str, Any] = json.loads(raw)
    return machine


def _check(machine: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(_core.fingerprint_check(json.dumps(machine)))
    return report


def test_a_recorded_snapshot_is_a_complete_fingerprint() -> None:
    m = _full_machine()
    report = _check(m)
    assert report["complete"] is True
    assert report["missing_required"] == []
    assert _core.fingerprint_is_complete(json.dumps(m)) is True


def test_a_missing_required_field_is_reported_and_fails_completeness() -> None:
    m = _full_machine()
    del m["l2_bytes"]
    m["sm_arch"] = None
    report = _check(m)
    assert report["complete"] is False
    assert set(report["missing_required"]) == {"l2_bytes", "sm_arch"}
    assert _core.fingerprint_is_complete(json.dumps(m)) is False


def test_framework_versions_are_recommended_not_required() -> None:
    m = _full_machine()
    m["toolkit"]["triton"] = None
    m["toolkit"]["torch"] = None
    report = _check(m)
    assert report["complete"] is True
    assert set(report["missing_recommended"]) == {"toolkit.triton", "toolkit.torch"}


def test_empty_machine_lists_every_required_field() -> None:
    report = _check({})
    assert report["complete"] is False
    assert len(report["missing_required"]) == 17


def test_fingerprint_check_rejects_non_json() -> None:
    with pytest.raises(ValueError):
        _core.fingerprint_check("{not json")


NVCC_OUT = (
    "nvcc: NVIDIA (R) Cuda compiler driver\n"
    "Copyright (c) 2005-2024 NVIDIA Corporation\n"
    "Cuda compilation tools, release 12.4, V12.4.131\n"
    "Build cuda_12.4.r12.4/compiler.34097967_0\n"
)
PTXAS_OUT = (
    "ptxas: NVIDIA (R) Ptx optimizing assembler\nCuda compilation tools, release 12.6, V12.6.20\n"
)


def test_toolchain_version_parsing() -> None:
    assert _core.parse_nvcc_version(NVCC_OUT) == "12.4.131"
    assert _core.parse_ptxas_version(PTXAS_OUT) == "12.6.20"
    assert _core.parse_nvcc_version("release 11.8, done") == "11.8"
    assert _core.parse_nvcc_version("no version here") is None


def test_toolchain_detect_shape_and_missing_tools() -> None:
    # On a machine with no CUDA / torch / triton every value is None, and the
    # probe must not raise.
    tc = _toolchain.detect()
    assert set(tc) == {"triton", "torch", "ptxas", "nvcc", "cuda_runtime"}
    for value in tc.values():
        assert value is None or isinstance(value, str)
    assert api.toolchain() == tc
