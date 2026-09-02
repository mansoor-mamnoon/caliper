"""Shared helpers for the ``do_bench`` misleads experiments.

Each experiment (``fast_kernel``, ``cold_warmup``, ``l2_resident``) measures one
kernel several ways and appends rows to ``docs/data/misleads.csv``:

    experiment,arch,method,value_us,note

The ``nsys`` column is filled by the reader -- every script prints the exact
``nsys`` command. Run on a CUDA host (Colab); ``make writeup-data`` runs all
three.

Nothing here imports ``torch`` at module load, so the scripts stay importable
(and lint-clean) on a machine with no GPU.
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "docs" / "data" / "misleads.csv"
FIELDS = ["experiment", "arch", "method", "value_us", "note"]


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - no torch on the dev box
        raise RuntimeError(
            "the misleads experiments need PyTorch with CUDA; run them on Colab"
        ) from exc
    if not torch.cuda.is_available():  # pragma: no cover - no CUDA on the dev box
        raise RuntimeError("no CUDA device; run the misleads experiments on Colab")
    return torch


def arch_tag() -> str:
    """``sm_XY`` for the current device."""
    major, minor = _torch().cuda.get_device_capability(0)
    return f"sm_{major}{minor}"


# --- the several ways to time a kernel -------------------------------------


def do_bench_median_us(fn: Any) -> float:
    """``caliper.do_bench`` with its Triton-matching defaults (warmup 25 ms),
    median, converted to microseconds. This is the number most people quote."""
    from caliper import do_bench

    ms = do_bench(fn, return_mode="median")
    assert not isinstance(ms, list)  # only `quantiles=` returns a list
    return ms * 1000.0


def caliper_steady_median_us(fn: Any, warmup_ms: float = 300.0, rep_ms: float = 600.0) -> float:
    """caliper's live per-launch samples, trimmed to steady state
    (``_core.steady_state_index``), median. A generous warmup budget plus the
    trim is what keeps a slow-ramp kernel honest."""
    from caliper import _core
    from caliper.api import live_timing_ms

    samples_ms = live_timing_ms(fn, warmup=warmup_ms, rep=rep_ms)
    start, _converged = _core.steady_state_index(samples_ms)
    trimmed = samples_ms[start:] or samples_ms
    return float(_core.summarize(trimmed)["p50"]) * 1000.0


def batched_launch_us(fn: Any, batch: int = 256) -> float:
    """One CUDA-event pair around ``batch`` back-to-back launches, divided by
    ``batch`` -- the measurement ``caliper.bench(batch=…)`` makes. Removes the
    per-launch ``cudaEventRecord`` overhead that inflates a short kernel."""
    torch = _torch()
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    per_launch_ms: list[float] = []
    for _ in range(30):
        start.record()
        for _ in range(batch):
            fn()
        end.record()
        torch.cuda.synchronize()
        per_launch_ms.append(start.elapsed_time(end) / batch)
    return statistics.median(per_launch_ms) * 1000.0


def naive_per_iter_sync_us(fn: Any, iters: int = 300) -> float:
    """The classic wrong loop: ``fn(); torch.cuda.synchronize()`` every
    iteration, timed on the host clock. The sync latency lands on every sample,
    so a 10 us kernel reads as 30-60 us."""
    torch = _torch()
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    per: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        per.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(per)


def no_flush_median_us(fn: Any, iters: int = 300) -> float:
    """A tight timed loop with **no** cache clear between reps. If the kernel's
    working set fits in L2, every rep after the first hits warm cache and the
    number is optimistic."""
    torch = _torch()
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return statistics.median(times_ms) * 1000.0


# --- output --------------------------------------------------------------


def nsys_command(script: str) -> str:
    return f"nsys profile --stats=true -o /tmp/{script} python examples/misleads/{script}.py --nsys"


def append_rows(rows: list[dict[str, Any]]) -> None:
    """Append ``rows`` (keyed by :data:`FIELDS`) to ``docs/data/misleads.csv``,
    writing the header if the file is new."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists() or not CSV_PATH.read_text().strip()
    with CSV_PATH.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def report(experiment: str, results: dict[str, float], note: str, *, write: bool) -> None:
    """Print a small table and (with ``write``) append it to the CSV."""
    arch = arch_tag()
    print(f"\n== {experiment} ({arch}) ==  {note}")
    for method, value in results.items():
        print(f"  {method:22} {value:8.2f} us")
    print(f"  nsys (fill me in):  {nsys_command(experiment)}")
    if write:
        append_rows(
            [
                {
                    "experiment": experiment,
                    "arch": arch,
                    "method": method,
                    "value_us": f"{value:.2f}",
                    "note": note,
                }
                for method, value in results.items()
            ]
        )
