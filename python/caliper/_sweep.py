"""``sweep()``: run a spec's cells, checkpoint after each, resume a killed run.

Each cell is one ``bench()`` call. After every cell the record is appended to a
``.state.jsonl`` sidecar and the output file is rewritten, so a killed sweep
resumes from exactly where it stopped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from caliper import _spec
from caliper._grid import Grid

__all__ = ["sweep"]

RunCell = Callable[[dict[str, Any]], dict[str, Any]]


def _safe(key: str) -> str:
    """A cell key as a filesystem-safe basename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("_")


def _opt_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _read_state(path: Path) -> dict[str, dict[str, Any]]:
    """The ``{key: record}`` from a ``.state.jsonl`` sidecar. A truncated final
    line (a mid-write kill) is skipped."""
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[entry["key"]] = entry["record"]
    return done


def _default_run_cell(recordings_dir: Path | None) -> RunCell:
    """A per-cell runner that replays a recording named after the cell key.
    Without ``recordings_dir`` there is nothing to run on a machine with no GPU."""

    def run(cell: dict[str, Any]) -> dict[str, Any]:
        if recordings_dir is None:
            raise NotImplementedError(
                "sweep() needs recordings_dir=<dir> (one <cell-key>.jsonl per cell) "
                "or a run_cell= override; the live launcher runs on a CUDA host."
            )
        keys = _spec.cell_keys([cell])
        rec = (recordings_dir / f"{_safe(keys[0])}.jsonl").read_text()
        from caliper.api import bench

        shape = cell["shape"]
        result = bench(
            cell["target"],
            recording=rec,
            dtype=cell["dtype"],
            shape={k.upper(): shape[k] for k in ("m", "n", "k") if k in shape},
            batches=int(cell["bench"]["min_samples"]),
        )
        record: dict[str, Any] = result.to_dict()
        return record

    return run


def sweep(
    spec: str | Path | dict[str, Any],
    *,
    recordings_dir: str | Path | None = None,
    run_cell: RunCell | None = None,
    parquet: str | Path | None = None,
    json_out: str | Path | None = None,
    resume: bool | None = None,
    state_path: str | Path | None = None,
) -> Grid:
    """Expand ``spec``, run every cell, and return the results as a :class:`Grid`.

    ``spec`` is a ``Path`` / YAML text / dict (Appendix D). Output goes to
    ``parquet`` and/or ``json_out`` (falling back to the spec's ``output:``
    block); at least one is required. After each cell the record is checkpointed
    to ``<output>.state.jsonl``; with ``resume=True`` (or the spec's
    ``output.resume``) a re-run skips the cells already recorded there.

    ``run_cell`` overrides how a cell is measured; the default replays a
    recording named ``<cell-key>.jsonl`` from ``recordings_dir``.
    """
    cells = _spec.load_cells(spec)
    keys = _spec.cell_keys(cells)
    out_block: dict[str, Any] = _spec.parse_spec(spec).get("output", {}) or {}

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

    pending = [c for c, k in zip(cells, keys, strict=True) if k not in done]
    if not do_resume and state.exists():
        state.unlink()  # a fresh (non-resume) run starts clean

    state.parent.mkdir(parents=True, exist_ok=True)
    with state.open("a") as fh:
        for cell in pending:
            key = _spec.cell_keys([cell])[0]
            record = runner(cell)
            done[key] = record
            fh.write(json.dumps({"key": key, "record": record}) + "\n")
            fh.flush()
            _write_outputs(done, keys, parquet_path, json_path)

    grid = Grid([done[k] for k in keys if k in done])
    _write_outputs(done, keys, parquet_path, json_path)
    return grid


def _write_outputs(
    done: dict[str, dict[str, Any]],
    keys: list[str],
    parquet_path: Path | None,
    json_path: Path | None,
) -> None:
    grid = Grid([done[k] for k in keys if k in done])
    if json_path is not None:
        grid.to_json(json_path, indent=1)
    if parquet_path is not None:
        grid.to_parquet(parquet_path)
