# caliper

**Measure GPU kernels correctly.**

Timing a GPU kernel looks like a one-liner and almost never is. The common
approaches get several things subtly wrong: they don't warm up to a steady
clock, they flush the wrong amount of cache (or none), they don't notice the
GPU throttling mid-measurement, they synchronise once per iteration (which
inflates anything short), and they report a single mean instead of a
distribution. The result is numbers that look precise and quietly disagree with
each other by 20-30%.

`caliper` is a small library and command-line tool that gets those details
right by default, and reports enough context alongside each number
(achieved vs. peak, register spills, occupancy, launch overhead, clock state)
that you can tell whether to trust it.

> **Status: early development.** This repository currently contains the project
> scaffold, the result schema, and continuous integration. The measurement
> engine is being built next. APIs and output formats will change until the
> first tagged release.

## What it will do

- **Time a kernel honestly** -- steady-state warmup, architecture-aware L2 cache
  flushing, GPU clock locking, thermal/power throttle detection with automatic
  discarding of bad samples, and a real latency distribution (p10 / p50 / p90 +
  spread) rather than one mean.
- **Handle short kernels** -- measure a batch of back-to-back launches between
  two events instead of synchronising after each one, so a 5-microsecond kernel
  isn't reported as 50.
- **Explain the number** -- achieved throughput vs. the hardware roofline for the
  dtype, arithmetic intensity, whether the kernel is compute- or
  memory-bound, register and shared-memory usage, spill counts, occupancy, and
  the CPU-side launch overhead.
- **Sweep a matrix** -- run one kernel across shapes, dtypes, memory layouts, and
  autotune configurations and write a stable, machine-readable results file.
- **Catch regressions** -- compare two results files and flag the ones that moved
  outside the measurement noise, with the register/occupancy delta that explains
  why.
- **Check itself** -- a `selftest` command validates an install against
  on-device reference workloads whose correct answers are known from first
  principles, and cross-checks against Nsight Systems where it's available.

## Planned interface

```python
from caliper import bench

result = bench(lambda: my_kernel(a, b))
print(result.p50_us, result.roofline_pct, result.ptxas.spill_stores_bytes)
```

```
caliper doctor          # is this machine set up to produce trustworthy numbers?
caliper bench k.py::fn   # measure one kernel
caliper sweep spec.yaml  # run a matrix -> results file
caliper compare --baseline a.parquet --candidate b.parquet
caliper selftest         # validate this install against on-device references
```

The full interface, data schema, and validation strategy are written up in
[`caliper-4-week-plan.md`](caliper-4-week-plan.md).

## Install

Not yet published. For development:

```bash
git clone https://github.com/mansoor-mamnoon/caliper
cd caliper
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The library targets Linux with an NVIDIA GPU (CUDA 12.1+) and Python
3.10-3.12. The pure computation and schema code has no GPU dependency and its
tests run anywhere.

## Development

```bash
make lint        # ruff
make typecheck   # mypy
make test        # pytest -m "l0 or l1"   (no GPU needed)
make check       # all of the above
```

Tests are tagged by what they need: `l0` (pure, no GPU), `l1` (against recorded
device responses, no GPU), and `l2`/`l3`/`l4`/`l6` (require a real GPU). CI runs
`l0` and `l1` on every push.

## License

MIT -- see [LICENSE](LICENSE).
