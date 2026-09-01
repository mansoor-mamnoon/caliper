"""L1: sweep() orchestration -- checkpoint after each cell, resume a killed run,
and time each autotune config once."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from caliper import Result, _spec, sweep

pytestmark = pytest.mark.l1

SPEC = {"target": "corpus:gemm", "dtypes": ["bf16", "fp16"], "shapes": "square-pow2"}
# 2 dtypes x 1 layout x 5 square-pow2 shapes = 10 cells


def _fake_record(cell: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    r = Result.default().to_dict()
    r["kernel"]["name"] = "corpus:gemm_bf16"
    r["kernel"]["dtype"] = cell["dtype"]
    r["kernel"]["layout"] = cell["layout"]
    r["kernel"]["source_hash"] = "src:gemm"
    r["kernel"]["autotune_config"] = config or {}
    r["machine"]["sm_arch"] = "sm_89"
    r["machine"]["driver"] = "550.90.07"
    base = float(cell["shape"]["m"])
    r["timing"]["p50_us"] = base - float((config or {}).get("BLOCK_M", 0)) / 10.0
    return r


def test_sweep_runs_every_cell_and_writes_outputs(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    seen: list[str] = []

    def run(cell: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        seen.append(_spec.cell_keys([cell])[0])
        return _fake_record(cell, config)

    pq = tmp_path / "out.parquet"
    grid = sweep(SPEC, run_cell=run, parquet=pq, json_out=tmp_path / "out.json")

    assert len(grid) == 10
    assert len(seen) == 10 and len(set(seen)) == 10
    assert pq.exists() and not (tmp_path / f"out.parquet.{__import__('os').getpid()}.tmp").exists()
    assert (tmp_path / "out.parquet.state.jsonl").exists()
    assert len(json.loads((tmp_path / "out.json").read_text())) == 10


def test_a_killed_sweep_resumes_and_does_not_rerun_cells(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pq = tmp_path / "out.parquet"
    calls_1: list[str] = []

    def kill_after_4(cell: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        calls_1.append(_spec.cell_keys([cell])[0])
        if len(calls_1) > 4:
            raise RuntimeError("runtime killed")
        return _fake_record(cell, config)

    with pytest.raises(RuntimeError, match="killed"):
        sweep(SPEC, run_cell=kill_after_4, parquet=pq)

    state = pq.with_suffix(pq.suffix + ".state.jsonl")
    assert len([ln for ln in state.read_text().splitlines() if ln.strip()]) == 4
    # a truncated final line (a real mid-write kill) must be tolerated on resume
    with state.open("a") as fh:
        fh.write('{"key": "trunc", "reco')

    calls_2: list[str] = []

    def finish(cell: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        calls_2.append(_spec.cell_keys([cell])[0])
        return _fake_record(cell, config)

    grid = sweep(SPEC, run_cell=finish, parquet=pq, resume=True)

    assert len(grid) == 10
    assert len(calls_2) == 6  # only the unfinished cells
    assert not set(calls_1[:4]) & set(calls_2)  # no cell run twice
    assert [r.kernel["dtype"] for r in grid] == ["bf16"] * 5 + ["fp16"] * 5  # spec order


def test_a_fresh_run_clears_a_stale_state_file(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pq = tmp_path / "out.parquet"
    state = pq.with_suffix(pq.suffix + ".state.jsonl")
    state.write_text('{"key": "stale", "record": {}}\n')
    n = {"count": 0}

    def run(cell: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        n["count"] += 1
        return _fake_record(cell, config)

    sweep(SPEC, run_cell=run, parquet=pq)
    assert n["count"] == 10
    keys = {json.loads(ln)["key"] for ln in state.read_text().splitlines() if ln.strip()}
    assert "stale" not in keys and len(keys) == 10


def test_autotune_configs_are_timed_once_and_the_fastest_kept(tmp_path: Path) -> None:
    """Second sweep adds one config -> only that config is re-timed (cache hit
    on the rest)."""
    one_cell = {
        "target": "corpus:gemm",
        "dtypes": ["bf16"],
        "shapes": [{"M": 4096, "N": 4096, "K": 4096}],
    }
    machine = {"sm_arch": "sm_89", "driver": "550.90.07", "cuda_runtime": "12.4", "toolkit": {}}
    cache = tmp_path / "autotune.json"
    timed: list[dict[str, Any]] = []

    def run(cell: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        timed.append(config)
        return _fake_record(cell, config)

    configs_v1 = [{"BLOCK_M": 64}, {"BLOCK_M": 128}, {"BLOCK_M": 256}]
    grid = sweep(
        one_cell,
        run_cell=run,
        configs_for=lambda _c: configs_v1,
        cache_path=cache,
        machine=machine,
        kernel_source_hash="src:gemm",
        json_out=tmp_path / "o.json",
    )
    assert len(timed) == 3  # all missed
    assert grid[0].kernel["autotune_config"] == {"BLOCK_M": 256}  # fastest (biggest subtraction)

    timed.clear()
    sweep(
        one_cell,
        run_cell=run,
        configs_for=lambda _c: [*configs_v1, {"BLOCK_M": 512}],
        cache_path=cache,
        machine=machine,
        kernel_source_hash="src:gemm",
        json_out=tmp_path / "o.json",
    )
    assert timed == [{"BLOCK_M": 512}]  # only the new config re-timed


def test_sweep_without_an_output_is_an_error() -> None:
    with pytest.raises(ValueError, match="output"):
        sweep(SPEC, run_cell=_fake_record)


def test_sweep_without_recordings_or_run_cell_is_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        sweep(SPEC, parquet=tmp_path / "o.parquet", json_out=tmp_path / "o.json")


def test_sweep_accepts_a_spec_filename_string(tmp_path: Path) -> None:
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("target: corpus:gemm\ndtypes: [bf16]\nshapes: square-pow2\n")
    grid = sweep(str(spec_file), run_cell=_fake_record, json_out=tmp_path / "o.json")
    assert len(grid) == 5
