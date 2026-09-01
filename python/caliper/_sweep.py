"""``sweep()``: run a spec's cells, checkpoint after each, resume a killed run.

Each cell is one or more ``bench()`` calls -- one per autotune config the kernel
exposes (``configs_for``), the fastest kept. After every cell the record is
appended to a ``.state.jsonl`` sidecar, so a killed sweep resumes from exactly
where it stopped. Per-config timings go through an :class:`AutotuneCache` when
``cache_path`` is set, so a re-sweep re-times only a newly-added config.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from caliper import _spec
from caliper._autotune import AutotuneCache
from caliper._grid import Grid

__all__ = ["sweep"]

#: measure one (cell, config) -> a record dict
RunCell = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
#: the autotune configs to time for a cell (``[{}]`` = no tuning)
ConfigsFor = Callable[[dict[str, Any]], list[dict[str, Any]]]


def _safe(key: str) -> str:
    """A cell key as a filesystem-safe basename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("_")


def _opt_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _read_state(path: Path) -> dict[str, dict[str, Any]]:
    """The ``{key: record}`` from a ``.state.jsonl`` sidecar. A truncated or
    malformed final line (a mid-write kill) is skipped."""
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            done[entry["key"]] = entry["record"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return done


def _no_tuning(_cell: dict[str, Any]) -> list[dict[str, Any]]:
    return [{}]


def _default_run_cell(recordings_dir: Path | None) -> RunCell:
    """Replay a recording named after the cell key. Without ``recordings_dir``
    there is nothing to run on a machine with no GPU."""

    def run(cell: dict[str, Any], _config: dict[str, Any]) -> dict[str, Any]:
        if recordings_dir is None:
            raise NotImplementedError(
                "sweep() needs recordings_dir=<dir> (one <cell-key>.jsonl per cell) "
                "or a run_cell= override; the live launcher runs on a CUDA host."
            )
        key = _spec.cell_keys([cell])[0]
        rec = (recordings_dir / f"{_safe(key)}.jsonl").read_text()
        from caliper.api import bench

        b = cell["bench"]
        shape = cell["shape"]
        result = bench(
            cell["target"],
            recording=rec,
            dtype=cell["dtype"],
            layout=cell["layout"],
            shape={
                k.upper(): v for k, v in shape.items() if k in ("m", "n", "k", "b", "h", "s", "d")
            },
            warmup=b["warmup"],
            cuda_graph=b["cuda_graph"],
            flush_l2=b["flush_l2"],
            lock_clocks=b["lock_clocks"],
        )
        record: dict[str, Any] = result.to_dict()
        return record

    return run


def _measure_cell(
    cell: dict[str, Any],
    run: RunCell,
    configs: list[dict[str, Any]],
    cache: AutotuneCache | None,
    machine: dict[str, Any] | None,
    ksh: str | None,
) -> dict[str, Any]:
    """Time every config for a cell (cache hits skipped) and return the record
    of the fastest one. The cache is only consulted once both the fingerprint
    (``machine``) and the kernel source hash (``ksh``) are known -- passed in,
    or discovered by timing the first config."""
    if len(configs) <= 1:
        return run(cell, configs[0] if configs else {})

    def keyed() -> AutotuneCache | None:
        return cache if (cache is not None and machine is not None and ksh is not None) else None

    best: tuple[float, dict[str, Any]] | None = None
    for config in configs:
        active = keyed()
        cached = active.get(machine, ksh, config) if active is not None else None  # type: ignore[arg-type]
        if cached is not None:
            p50, record = cached["p50_us"], cached["record"]
        else:
            record = run(cell, config)
            p50 = float(record["timing"]["p50_us"] or 0.0)
            machine = machine or (record.get("machine") or {})
            ksh = ksh or (record.get("kernel") or {}).get("source_hash") or cell["target"]
            active = keyed()
            if active is not None:
                active.put(machine, ksh, config, {"p50_us": p50, "record": record})
        if best is None or p50 < best[0]:
            best = (p50, record)

    assert best is not None
    return best[1]


def sweep(
    spec: str | Path | dict[str, Any],
    *,
    recordings_dir: str | Path | None = None,
    run_cell: RunCell | None = None,
    configs_for: ConfigsFor | None = None,
    cache_path: str | Path | None = None,
    machine: dict[str, Any] | None = None,
    kernel_source_hash: str | None = None,
    parquet: str | Path | None = None,
    json_out: str | Path | None = None,
    resume: bool | None = None,
    state_path: str | Path | None = None,
) -> Grid:
    """Expand ``spec``, run every cell, and return the results as a :class:`Grid`.

    ``spec`` is a dict / ``Path`` / filename / YAML text (Appendix D). Output
    goes to ``parquet`` and/or ``json_out`` (falling back to the spec's
    ``output:`` block); at least one is required. After each cell the record is
    checkpointed to ``<output>.state.jsonl``; with ``resume=True`` (or the
    spec's ``output.resume``) a re-run skips the cells already recorded there.

    ``configs_for(cell)`` returns the autotune configs to time for a cell
    (default: one empty config -- no tuning). ``run_cell(cell, config)``
    overrides how a (cell, config) is measured; the default replays a
    ``<cell-key>.jsonl`` recording from ``recordings_dir``. When ``cache_path``
    is set, per-config timings are cached there so a re-sweep re-times only a
    newly-added config; pass ``machine=`` the fingerprint so even the first
    config can be served from the cache.
    """
    spec_dict = _spec.parse_spec(spec)
    cells = _spec.load_cells(spec_dict)
    keys = _spec.cell_keys(cells)
    out_block: dict[str, Any] = spec_dict.get("output", {}) or {}

    parquet_path = Path(parquet) if parquet else _opt_path(out_block.get("parquet"))
    json_path = Path(json_out) if json_out else _opt_path(out_block.get("json"))
    if parquet_path is None and json_path is None:
        raise ValueError(
            "sweep() needs an output: pass parquet= / json_out= or set output: in the spec"
        )

    primary = parquet_path or json_path
    assert primary is not None
    state = Path(state_path) if state_path else primary.with_suffix(primary.suffix + ".state.jsonl")

    do_resume = out_block.get("resume", False) if resume is None else resume
    done = _read_state(state) if do_resume else {}

    runner = run_cell or _default_run_cell(Path(recordings_dir) if recordings_dir else None)
    pick_configs = configs_for or _no_tuning
    cache = AutotuneCache(cache_path) if cache_path is not None else None

    if not do_resume and state.exists():
        state.unlink()  # a fresh run starts clean
    state.parent.mkdir(parents=True, exist_ok=True)

    with state.open("a") as fh:
        for cell, key in zip(cells, keys, strict=True):
            if key in done:
                continue
            record = _measure_cell(
                cell, runner, pick_configs(cell), cache, machine, kernel_source_hash
            )
            done[key] = record
            fh.write(json.dumps({"key": key, "record": record}) + "\n")
            fh.flush()

    grid = Grid([done[k] for k in keys if k in done])
    _write_outputs(grid, parquet_path, json_path)
    return grid


def _atomic(path: Path, write: Callable[[Path], Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    write(tmp)
    tmp.replace(path)


def _write_outputs(grid: Grid, parquet_path: Path | None, json_path: Path | None) -> None:
    """Write the grid to its output(s), each via a temp sibling + atomic rename."""
    if json_path is not None:
        _atomic(json_path, lambda p: grid.to_json(p, indent=1))
    if parquet_path is not None:
        _atomic(parquet_path, grid.to_parquet)
