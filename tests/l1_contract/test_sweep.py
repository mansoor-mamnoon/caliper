"""L1: sweep() orchestration -- checkpoint after each cell, resume a killed run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from caliper import Result, _spec, sweep

pytestmark = pytest.mark.l1

SPEC = {"target": "corpus:gemm", "dtypes": ["bf16", "fp16"], "shapes": "square-pow2"}
# 2 dtypes x 1 layout x 5 square-pow2 shapes = 10 cells


def _fake_record(cell: dict[str, Any]) -> dict[str, Any]:
    r = Result.default().to_dict()
    r["kernel"]["name"] = "corpus:gemm_bf16"
    r["kernel"]["dtype"] = cell["dtype"]
    r["timing"]["p50_us"] = float(cell["shape"]["m"])
    return r


def test_sweep_runs_every_cell_and_writes_outputs(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    seen: list[str] = []

    def run(cell: dict[str, Any]) -> dict[str, Any]:
        seen.append(_spec.cell_keys([cell])[0])
        return _fake_record(cell)

    pq = tmp_path / "out.parquet"
    grid = sweep(SPEC, run_cell=run, parquet=pq, json_out=tmp_path / "out.json")

    assert len(grid) == 10
    assert len(seen) == 10 and len(set(seen)) == 10
    assert pq.exists()
    assert (tmp_path / "out.parquet.state.jsonl").exists()
    # the JSON output is a valid Grid of 10 records
    reload = json.loads((tmp_path / "out.json").read_text())
    assert len(reload) == 10


def test_a_killed_sweep_resumes_and_does_not_rerun_cells(tmp_path: Path) -> None:
    pq = tmp_path / "out.parquet"
    pytest.importorskip("pyarrow")

    calls_1: list[str] = []

    def kill_after_4(cell: dict[str, Any]) -> dict[str, Any]:
        calls_1.append(_spec.cell_keys([cell])[0])
        if len(calls_1) > 4:
            raise RuntimeError("runtime killed")
        return _fake_record(cell)

    with pytest.raises(RuntimeError, match="killed"):
        sweep(SPEC, run_cell=kill_after_4, parquet=pq)

    state = pq.with_suffix(pq.suffix + ".state.jsonl")
    assert len([ln for ln in state.read_text().splitlines() if ln.strip()]) == 4

    calls_2: list[str] = []

    def finish(cell: dict[str, Any]) -> dict[str, Any]:
        calls_2.append(_spec.cell_keys([cell])[0])
        return _fake_record(cell)

    grid = sweep(SPEC, run_cell=finish, parquet=pq, resume=True)

    assert len(grid) == 10
    assert len(calls_2) == 6  # only the unfinished cells
    assert not set(calls_1[:4]) & set(calls_2)  # no cell run twice
    # cell order is preserved in the final grid
    dtypes = [r.kernel["dtype"] for r in grid]
    assert dtypes == ["bf16"] * 5 + ["fp16"] * 5


def test_a_fresh_run_clears_a_stale_state_file(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pq = tmp_path / "out.parquet"
    state = pq.with_suffix(pq.suffix + ".state.jsonl")
    state.write_text('{"key": "stale", "record": {}}\n')

    n = {"count": 0}

    def run(cell: dict[str, Any]) -> dict[str, Any]:
        n["count"] += 1
        return _fake_record(cell)

    sweep(SPEC, run_cell=run, parquet=pq)  # no resume -> ignores + clears the stale state
    assert n["count"] == 10
    keys = {json.loads(ln)["key"] for ln in state.read_text().splitlines() if ln.strip()}
    assert "stale" not in keys and len(keys) == 10


def test_sweep_without_an_output_is_an_error() -> None:
    with pytest.raises(ValueError, match="output"):
        sweep(SPEC, run_cell=_fake_record)


def test_sweep_without_recordings_or_run_cell_is_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        sweep(SPEC, parquet=tmp_path / "o.parquet", json_out=tmp_path / "o.json")
