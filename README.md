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
> scaffold: a Rust measurement core, its Python bindings, the result schema, and
> continuous integration. The measurement engine is being built next. APIs and
> output formats will change until the first tagged release.

## What it will do

- **Time a kernel honestly** -- steady-state warmup, architecture-aware L2 cache
  flushing, GPU clock locking, thermal/power throttle detection with automatic
  discarding of bad samples, and a real latency distribution (p10 / p50 / p90 +
  spread) rather than one mean.
- **Handle short kernels** -- measure a batch of back-to-back launches between
  two events instead of synchronising after each one, so a 5-microsecond kernel
  isn't reported as 50.
- **Explain the number** -- achieved throughput vs. the hardware roofline for the
  dtype, arithmetic intensity, whether the kernel is compute- or memory-bound,
  register and shared-memory usage, spill counts, occupancy, and the CPU-side
  launch overhead.
- **Sweep a matrix** -- run one kernel across shapes, dtypes, memory layouts, and
  autotune configurations and write a stable, machine-readable results file.
- **Catch regressions** -- compare two results files and flag the ones that moved
  outside the measurement noise, with the register/occupancy delta that explains
  why.
- **Check itself** -- a `selftest` command validates an install against
  on-device reference workloads whose correct answers are known from first
  principles, and cross-checks against Nsight Systems where it's available.

## How it's built

| Layer | Language | What lives here |
| --- | --- | --- |
| `crates/caliper-core` | Rust | All measurement logic: the result schema, statistics, the roofline model, `ptxas` parsing, the regression threshold model. No GPU or Python dependency; tested with `cargo test`. |
| `crates/caliper-ffi` | Rust (PyO3) | A thin binding layer that exposes the core to Python as `caliper._core`. |
| GPU layer *(coming)* | Rust + a small amount of CUDA C++ | Kernel launch, CUDA events and graphs, NVML clock control, and the on-device reference kernels. |
| `python/caliper` | Python | The public API, the command-line tool, and the Triton-compatible `do_bench` shim. |

The full interface, data schema, and validation strategy are written up in
[`docs/plan.md`](docs/plan.md).

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

## Install

Not yet published. For development you need a Rust toolchain
([rustup](https://rustup.rs)) and Python 3.10-3.12:

```bash
git clone https://github.com/mansoor-mamnoon/caliper
cd caliper
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # builds the Rust extension via maturin
```

The library targets Linux with an NVIDIA GPU (CUDA 12.1+). The core and its
tests have no GPU dependency and run anywhere.

## Development

```bash
make check     # cargo fmt + clippy + cargo test, then ruff + mypy + pytest
```

or individually:

```bash
make rust-test    # cargo test --all
make test         # pytest -m "l0 or l1"   (no GPU needed)
make lint         # ruff
make typecheck    # mypy
```

Tests are tagged by what they need: `l0` (pure, no GPU), `l1` (against recorded
device responses, no GPU), and `l2`/`l3`/`l4`/`l6` (require a real GPU). CI runs
the Rust suite and the `l0`/`l1` Python tests on every push.

## License

MIT -- see [LICENSE](LICENSE).
