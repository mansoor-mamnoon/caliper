"""L0: the Grid table -- JSON / filter always, Parquet when pyarrow is present."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from caliper import Grid, Result

pytestmark = pytest.mark.l0


def _record(sm_arch: str, p50: float) -> dict[str, Any]:
    r = Result.default().to_dict()
    r["machine"]["sm_arch"] = sm_arch
    r["machine"]["driver"] = "550.90.07"
    r["machine"]["toolkit"] = {"triton": "3.2.0", "torch": "2.6.0"}
    r["timing"]["p50_us"] = p50
    r["kernel"]["name"] = "matmul"
    return r


def _grid() -> Grid:
    return Grid([_record("sm_89", 200.0), _record("sm_90", 120.0), _record("sm_89", 240.0)])


def test_len_iter_and_index() -> None:
    g = _grid()
    assert len(g) == 3
    assert [r.machine["sm_arch"] for r in g] == ["sm_89", "sm_90", "sm_89"]
    assert g[1].p50_us == 120.0


def test_filter_returns_a_new_grid() -> None:
    g = _grid()
    ada = g.filter(lambda r: r.machine["sm_arch"] == "sm_89")
    assert isinstance(ada, Grid)
    assert len(ada) == 2
    assert len(g) == 3  # original untouched


def test_json_round_trip(tmp_path: Path) -> None:
    g = _grid()
    p = tmp_path / "grid.json"
    text = g.to_json(p, indent=1)
    assert p.read_text() == text
    back = Grid.from_json(p)
    assert [r.to_dict() for r in back] == [r.to_dict() for r in g]
    assert [r.to_dict() for r in Grid.from_json(text)] == [r.to_dict() for r in g]


def test_from_json_rejects_a_non_array() -> None:
    with pytest.raises(ValueError, match="array"):
        Grid.from_json('{"not": "an array"}')


def test_parquet_round_trip_and_schema(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    g = _grid()
    p = tmp_path / "grid.parquet"
    g.to_parquet(p)

    schema_names = set(pq.read_schema(p).names)
    assert "timing.p50_us" in schema_names
    assert "machine.sm_arch" in schema_names
    assert "ptxas.regs_per_thread" in schema_names
    assert "toolchain_hash" in schema_names  # Appendix C derived column

    table = pq.read_table(p)
    assert table.num_rows == 3
    hashes = table.column("toolchain_hash").to_pylist()
    assert hashes[0] == hashes[2]  # same toolkit + driver
    # a bare sha256 hex digest -- it becomes a directory segment in caliper-results
    assert all(isinstance(h, str) and len(h) == 64 and int(h, 16) >= 0 for h in hashes)

    back = Grid.from_parquet(p)
    assert [r.machine["sm_arch"] for r in back] == ["sm_89", "sm_90", "sm_89"]
    assert [r.p50_us for r in back] == [200.0, 120.0, 240.0]


def test_written_parquet_validates(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    from caliper import api

    p = tmp_path / "grid.parquet"
    _grid().to_parquet(p)
    report = api.validate_records(p)
    assert report["ok"] is True
    assert report["n"] == 3


def test_json_grid_with_a_bad_row_is_reported(tmp_path: Path) -> None:
    from caliper import api

    good = _record("sm_89", 200.0)
    bad = _record("sm_89", 200.0)
    bad["timing"]["p50_us"] = -5.0  # negative time
    p = tmp_path / "g.json"
    p.write_text(json.dumps([good, bad]))
    report = api.validate_records(p)
    assert report["ok"] is False
    assert report["n_invalid"] == 1
    assert report["problems"][0]["row"] == 1
