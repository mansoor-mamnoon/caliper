"""L0: the Rust ptxas / cuobjdump / HIP parser, through the bindings.

Exhaustive coverage is in `cargo test`; these check the binding surface and a
couple of anchor values against the same golden captures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from caliper import _core

pytestmark = pytest.mark.l0

CAPTURES = Path(__file__).resolve().parents[2] / "crates" / "caliper-core" / "tests" / "ptxas"


def _parse(name: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = json.loads(_core.parse_ptxas((CAPTURES / name).read_text()))
    return result


def test_ptxas_verbose_with_spills() -> None:
    (k,) = _parse("spills_smem.txt")
    assert k["name"] == "matmul_kernel_0d1d2d3de4de5de"
    assert k["target"] == "sm_90a"
    assert k["ptxas"]["regs_per_thread"] == 168
    assert k["ptxas"]["spill_stores_bytes"] == 48
    assert k["ptxas"]["spill_loads_bytes"] == 32
    assert k["ptxas"]["smem_static_bytes"] == 99328
    assert k["ptxas"]["smem_dynamic_bytes"] is None


def test_multi_kernel_module() -> None:
    ks = _parse("multi.txt")
    assert [k["name"] for k in ks] == ["fwd_kernel", "bwd_kernel"]
    assert ks[1]["ptxas"]["regs_per_thread"] == 255


def test_cuobjdump_res_usage() -> None:
    ks = _parse("cuobjdump.txt")
    assert ks[0]["ptxas"]["regs_per_thread"] == 10
    assert ks[1]["ptxas"]["local_bytes"] == 16
    assert ks[1]["ptxas"]["spill_stores_bytes"] is None  # not in -res-usage


def test_hip_amdgpu() -> None:
    (k,) = _parse("hip.txt")
    assert k["ptxas"]["regs_per_thread"] == 40  # NumVgprs
    assert k["sgprs"] == 36
    assert k["ptxas"]["smem_static_bytes"] == 8192  # LDSByteSize


@pytest.mark.parametrize("bad", ["", "   \n", "this is not compiler output\nno structure here"])
def test_malformed_input_raises(bad: str) -> None:
    with pytest.raises(ValueError):
        _core.parse_ptxas(bad)
