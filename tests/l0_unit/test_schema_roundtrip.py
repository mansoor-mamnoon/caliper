"""L0: the Result schema must round-trip through dict and JSON without loss."""

from __future__ import annotations

import json

import pytest

from caliper import KernelLabel, Result
from caliper._internal.schema import SCHEMA_VERSION, Machine, Toolkit

pytestmark = pytest.mark.l0


def test_default_result_has_versions_and_sections() -> None:
    r = Result()
    assert r.schema_version == SCHEMA_VERSION
    assert r.caliper_version  # non-empty
    # every section object exists so callers can assign into it
    assert isinstance(r.kernel, KernelLabel)
    assert isinstance(r.machine, Machine)
    assert isinstance(r.machine.toolkit, Toolkit)
    assert r.throttle_reasons == []
    assert r.flags == []


def test_default_result_roundtrips_through_dict() -> None:
    r = Result()
    assert Result.from_dict(r.to_dict()) == r


def test_populated_result_roundtrips_through_json() -> None:
    r = Result(measured_at="2026-01-02T03:04:05Z", host_id_class="sha256:abc")
    r.kernel = KernelLabel(
        name="matmul_kernel",
        impl="triton",
        source_hash="sha256:deadbeef",
        autotune_config={"BLOCK_M": 128, "num_warps": 8},
        dtype="bf16",
        shape={"M": 4096, "N": 4096, "K": 4096},
        layout="row",
    )
    r.timing.p10_us = 241.0
    r.timing.p50_us = 243.2
    r.timing.p90_us = 250.1
    r.timing.mad_us = 1.4
    r.timing.n_samples = 300
    r.roofline.achieved_tflops = 565.0
    r.roofline.roofline_pct = 0.86
    r.roofline.bound = "compute"
    r.ptxas.regs_per_thread = 168
    r.ptxas.spill_stores_bytes = 0
    r.occupancy.theoretical = 0.25
    r.occupancy.achieved = 0.247
    r.clocks.sm_mhz = 2520
    r.clocks.locked = True
    r.clocks.lock_method = "nvml"
    r.machine.gpu_name = "NVIDIA GeForce RTX 4090"
    r.machine.sm_arch = "sm_89"
    r.machine.l2_bytes = 75_497_472
    r.machine.toolkit = Toolkit(triton="3.2.0", torch="2.6.0", ptxas="12.4.131")
    r.throttle_reasons = ["SW_POWER_CAP"]
    r.flags = ["clocks-unlocked"]

    as_dict = r.to_dict()
    text = json.dumps(as_dict)
    rebuilt = Result.from_dict(json.loads(text))

    assert rebuilt == r
    assert rebuilt.to_dict() == as_dict


def test_from_dict_ignores_unknown_keys_and_missing_sections() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kernel": {"name": "k", "not_a_field": 1},
        "unexpected_top_level": 42,
    }
    r = Result.from_dict(payload)
    assert r.kernel.name == "k"
    assert r.timing.p50_us is None  # missing section -> default


def test_from_dict_tolerates_missing_version() -> None:
    r = Result.from_dict({"kernel": {"impl": "cuda"}})
    assert r.schema_version == SCHEMA_VERSION
    assert r.kernel.impl == "cuda"
