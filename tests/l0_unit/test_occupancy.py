"""L0: the Rust theoretical-occupancy model, through the bindings.

Exhaustive coverage (every reference-table row) is in `cargo test`; these check
the binding surface and re-verify a few rows of the same checked-in CUDA
Occupancy Calculator reference table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from caliper import _core

pytestmark = pytest.mark.l0

REFERENCE = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "caliper-core"
    / "tests"
    / "occupancy"
    / "reference.csv"
)


def _occ(arch: str, regs: int, smem: int, block: int) -> dict[str, Any]:
    raw = _core.theoretical_occupancy(arch, regs, smem, block)
    assert raw is not None
    result: dict[str, Any] = json.loads(raw)
    return result


def _reference_rows() -> list[tuple[str, int, int, int, float, int, str]]:
    rows: list[tuple[str, int, int, int, float, int, str]] = []
    for line in REFERENCE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split(",")
        rows.append((f[0], int(f[1]), int(f[2]), int(f[3]), float(f[4]), int(f[5]), f[6]))
    return rows


def test_reference_table_has_at_least_ten_rows() -> None:
    assert len(_reference_rows()) >= 10


@pytest.mark.parametrize("row", _reference_rows())
def test_model_matches_the_occupancy_calculator_reference(
    row: tuple[str, int, int, int, float, int, str],
) -> None:
    arch, regs, smem, block, want_occ, want_warps, want_limiter = row
    got = _occ(arch, regs, smem, block)
    assert got["theoretical"] == pytest.approx(want_occ, abs=1e-9)
    assert got["active_warps_per_sm"] == want_warps
    assert got["limiter"] == want_limiter


def test_doubling_registers_halves_occupancy() -> None:
    full = _occ("sm_80", 32, 0, 256)
    half = _occ("sm_80", 64, 0, 256)
    assert full["theoretical"] == pytest.approx(1.0)
    assert half["theoretical"] == pytest.approx(0.5)


def test_unknown_arch_returns_none() -> None:
    assert _core.theoretical_occupancy("sm_42", 32, 0, 256) is None


@pytest.mark.parametrize(
    ("regs", "block"),
    [(32, 0), (32, 2048), (300, 256), (0, 256)],
)
def test_out_of_range_launch_config_raises(regs: int, block: int) -> None:
    with pytest.raises(ValueError):
        _core.theoretical_occupancy("sm_80", regs, 0, block)
