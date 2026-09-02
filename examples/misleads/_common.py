"""Shared helpers for the ``do_bench`` misleads experiments.

Each experiment (``cold_warmup``, ``fast_kernel``, ``l2_resident``) measures one
kernel several ways and returns rows shaped like ``docs/data/misleads.csv``:

    experiment,arch,method,value_us,note

Every experiment also emits an empty ``method=nsys`` row for the reader to fill
from ``nsys stats`` (the command is printed). ``make writeup-data``
(``run_all.py``) runs all three and rewrites the CSV atomically.

Only ``run_all`` writes the CSV; a single script just prints its table. Nothing
here imports ``torch`` at module load, so the scripts stay importable (and
lint-clean) on a machine with no GPU.
"""

from __future__ import annotations

import csv
import os
import statistics
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "docs" / "data" / "misleads.csv"
FIELDS = ["experiment", "arch", "method", "value_us", "note"]

#: the ``method`` values each experiment reports, in order (the CSV template and
#: the writeup tables must match this).
METHODS: dict[str, tuple[str, ...]] = {
    "cold_warmup": ("do_bench_warmup_25ms", "caliper_steady_state", "nsys"),
    "fast_kernel": ("naive_per_iter_sync", "do_bench_default", "caliper_batched", "nsys"),
    "l2_resident": ("no_flush", "do_bench_flushed", "nsys"),
}
#: the order run_all measures them in -- cold_warmup first so its device is
#: actually cold (see cold_warmup.py).
ORDER = ("cold_warmup", "fast_kernel", "l2_resident")


def require_cuda() -> Any:
    """Return the ``torch`` module, or raise a clear ``RuntimeError`` telling
    the reader to run on Colab. Every GPU path -- helpers and the scripts'
    ``kernel()`` / ``--nsys`` branches -- goes through this."""
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
    major, minor = require_cuda().cuda.get_device_capability(0)
    return f"sm_{major}{minor}"


# --- the several ways to time a kernel -------------------------------------


def do_bench_median_us(fn: Any) -> float:
    """``caliper.do_bench`` with its Triton-matching defaults (warmup 25 ms),
    median, converted to microseconds. This is the number most people quote."""
    from caliper import do_bench

    ms = do_bench(fn, return_mode="median")
    if isinstance(ms, list):  # only `quantiles=` returns a list
        raise TypeError("do_bench(return_mode='median') should return a scalar")
    return ms * 1000.0


def caliper_steady_median_us(fn: Any, warmup_ms: float = 300.0, rep_ms: float = 600.0) -> float:
    """caliper's live per-launch samples, trimmed to steady state
    (``caliper._core.steady_state_index``), median. A generous warmup budget
    plus the trim is what keeps a slow-ramp kernel honest. These are shipped
    caliper primitives; the full clock-locked ``caliper bench`` pipeline is not
    wired to a live launcher yet."""
    from caliper import _core
    from caliper.api import live_timing_ms

    samples_ms = live_timing_ms(fn, warmup=warmup_ms, rep=rep_ms)
    start, _converged = _core.steady_state_index(samples_ms)
    trimmed = samples_ms[start:] or samples_ms
    return float(_core.summarize(trimmed)["p50"]) * 1000.0


def batched_launch_us(fn: Any, batch: int = 256) -> float:
    """One CUDA-event pair around ``batch`` back-to-back launches, divided by
    ``batch`` -- the batched shape ``caliper bench(batch=…)`` uses on-device (a
    live launcher for it is still pending). Removes the per-launch
    ``cudaEventRecord`` overhead that inflates a short kernel."""
    torch = require_cuda()
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
    torch = require_cuda()
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
    torch = require_cuda()
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


def report(experiment: str, results: dict[str, float], note: str) -> list[dict[str, str]]:
    """Print the table and return the CSV rows -- one per measured method plus
    an empty ``nsys`` row for the reader to fill."""
    arch = arch_tag()
    print(f"\n== {experiment} ({arch}) ==  {note}")
    for method, value in results.items():
        print(f"  {method:22} {value:8.2f} us")
    print(f"  nsys (fill me in):  {nsys_command(experiment)}")

    rows = [
        {"experiment": experiment, "arch": arch, "method": m, "value_us": f"{v:.2f}", "note": note}
        for m, v in results.items()
    ]
    rows.append(
        {
            "experiment": experiment,
            "arch": arch,
            "method": "nsys",
            "value_us": "",
            "note": "fill from `nsys stats`",
        }
    )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    """Rewrite ``docs/data/misleads.csv`` atomically from ``rows`` (temp file +
    ``os.replace``), so a failed / half-run leaves the committed file intact."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CSV_PATH.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})
    os.replace(tmp, CSV_PATH)
