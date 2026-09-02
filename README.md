# caliper

**Measure GPU kernels correctly.**

[![CI](https://github.com/mansoor-mamnoon/caliper/actions/workflows/ci-cpu.yml/badge.svg)](https://github.com/mansoor-mamnoon/caliper/actions/workflows/ci-cpu.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mansoor-mamnoon/caliper/blob/main/notebooks/quickstart.ipynb)

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

For a concrete walk through four ways a default `do_bench` call misleads --
with reproduction scripts -- see
[**Your Triton benchmark is probably lying to you**](docs/why-do_bench-misleads.md).

**30 seconds** — no GPU needed:

```bash
pip install -e ".[dev]"     # from a clone (not on PyPI yet); builds the Rust core
caliper doctor              # is this box fit to benchmark?
caliper bench k --recording crates/caliper-gpu/fixtures/bench/happy.jsonl --batches 40 --json
```

Or [**open the quickstart in Colab**](https://colab.research.google.com/github/mansoor-mamnoon/caliper/blob/main/notebooks/quickstart.ipynb).

> **Status: early development.** The measurement engine, result schema, roofline
> and occupancy models, `ptxas` parsing, the sweep planner, the regression
> `compare`, the `submit` bundle + validator, the `do_bench` shim, and the
> command-line tool are built and tested. The on-device launcher for the
> CUDA-C++ oracle kernels (real CUDA events, NVML clock locking) is still a
> stub, so `bench()` runs against recorded device sessions for now -- no GPU
> required.
> The reference kernel corpus (`gemm`, `rmsnorm`, `softmax`, `attention_fwd`,
> `attention_bwd` -- Triton, not CUDA C++) doesn't go through that launcher, so
> it runs without waiting on it -- on any CUDA host, with on-device
> verification on Colab like the other GPU tiers; see
> [`docs/corpus.md`](docs/corpus.md). APIs and output formats will change until
> the first tagged release.

## What it does

- **Times a kernel honestly** -- steady-state warmup, architecture-aware L2 cache
  flushing, GPU clock locking, thermal/power throttle detection with automatic
  discarding of bad samples, and a real latency distribution (p10 / p50 / p90,
  mean, min/max, spread) rather than one mean.
- **Handles short kernels** -- measures a batch of back-to-back launches between
  two events instead of synchronising after each one, so a 5-microsecond kernel
  isn't reported as 50; `cuda_graph="auto"` captures the batch into a graph when
  the launch overhead would otherwise dominate.
- **Explains the number** -- achieved throughput vs. the hardware roofline for the
  dtype (a cited per-architecture peaks table), arithmetic intensity, whether the
  kernel is compute-, memory-, or latency-bound, register and shared-memory
  usage, spill counts, theoretical occupancy, and the CPU-side launch overhead.
- **Sweeps a matrix** -- runs one kernel across shapes, dtypes, and memory
  layouts (named shape libraries or an inline list), checkpoints after every
  cell, and resumes a killed run without re-measuring what finished. Results go
  to a stable Parquet or JSON file.
- **Catches regressions** -- `caliper compare` diffs two results files facet by
  facet against a variance-aware noise band (from the baseline's own MAD, or an
  explicit `--threshold`), so a real slowdown fires but run-to-run jitter
  doesn't. It prints the `ptxas` and occupancy deltas for a moved facet, flags a
  register-spill increase (which fails the run even under an explicit
  `--threshold`), and lists autotune configs that stopped being timed;
  `--fail-on-regression` makes it exit 1 for CI.
- **Checks itself** -- `caliper selftest` runs on-device reference workloads
  whose correct answers are known from first principles and reports `PASS` /
  `FAIL` / `ERROR` with a coverage note; `caliper validate` checks a results file
  against the schema, or a submission bundle against the shared results gate.
- **Shares your GPU's numbers** -- `caliper submit` packs a results file into a
  bundle (a manifest with the architecture, toolchain hash, clock-lock tier,
  and determinism / calibration checks, plus the rows and the machine
  fingerprint) that the community [`caliper-results`](results-repo/) repo's CI
  validates on every PR -- the same gate `caliper validate <bundle>` runs
  locally.
- **Is Triton-compatible** -- `caliper.do_bench` matches `triton.testing.do_bench`
  argument for argument, so a script can swap the import; the failure modes of a
  default `do_bench` call, and the reproduction scripts, are written up in
  [`docs/why-do_bench-misleads.md`](docs/why-do_bench-misleads.md).
- **Ships a reference kernel corpus** -- `gemm`, `rmsnorm`, `softmax`,
  `attention_fwd`, `attention_bwd`, each a Triton implementation pinned to a
  content hash plus a vendor baseline (cuBLAS / torch / SDPA), timed live on
  any CUDA host and checked against the same roofline model (and, for
  attention, an `allclose` against the baseline). See
  [`docs/corpus.md`](docs/corpus.md).

## How it's built

| Layer | Language | What lives here |
| --- | --- | --- |
| `crates/caliper-core` | Rust | All measurement logic: the result schema, statistics, steady-state detection, the reduction pipeline, the roofline and occupancy models, `ptxas` / `cuobjdump` / HIP parsing, the sweep spec parser, the autotune cache key, the regression threshold model, the submit-bundle manifest and validator, the oracle checks and the `selftest` report. No GPU or Python dependency; tested with `cargo test`. |
| `crates/caliper-gpu` | Rust | The device layer: four ports (launch, clocks, device info, module probe), a fixture player that replays a recorded session with no GPU, a recorder, and the feature-gated real CUDA/NVML implementations. |
| `crates/caliper-ffi` | Rust (PyO3) | A thin binding layer that exposes the core to Python as `caliper._core`. |
| `python/caliper` | Python | The public API, the command-line tool, the `do_bench` shim, YAML/Parquet I/O, and orchestration (`sweep`, `compare`). |
| `python/caliper/corpus` | Python + Triton | The reference kernel corpus (`gemm`, `rmsnorm`, `softmax`, `attention_fwd`, `attention_bwd`) and its vendor baselines -- runs live on any CUDA host, independent of the (still-stubbed) Rust launcher. See [`docs/corpus.md`](docs/corpus.md). |
| `crates/caliper-gpu/kernels` *(Colab)* | CUDA C++ | The on-device oracle kernels O1-O7. |

Full reference: [`docs/api.md`](docs/api.md) (every public symbol + the record
schema), [`docs/cli.md`](docs/cli.md) (every command + exit codes),
[`docs/shapes.md`](docs/shapes.md) (the named shape libraries). The design doc
is [`docs/plan.md`](docs/plan.md).

## Interface

```python
from caliper import bench, compare, do_bench, submit, sweep

# one kernel, from a recorded device session (a live callable needs a CUDA host)
result = bench(
    "corpus:gemm",
    recording=open("session.jsonl").read(),
    shape={"M": 4096, "N": 4096, "K": 4096},
    dtype="bf16",
)
print(result.p50_us, result.roofline_pct, result.ptxas.spill_stores_bytes)

# a Triton script only changes its import
ms = do_bench(fn, quantiles=[0.5, 0.2, 0.8])

# a matrix -> Parquet, resumable
from pathlib import Path

grid = sweep(Path("spec.yaml"))

# diff two results files for variance-aware regressions
report = compare("base.parquet", "candidate.parquet", fail_on_regression=True)
print(report["summary"], report["any_regression"])

# pack a bundle for the community results repo
submit("results.parquet", out="bundle/")

# a reference kernel, timed live (needs a CUDA host + `pip install 'caliper-gpu[triton]'`)
from caliper.corpus.kernels import attention_fwd, gemm

result = gemm.run({"shape": {"m": 4096, "n": 4096, "k": 4096}, "dtype": "bf16", "layout": "row"})
attn_cell = {"shape": {"B": 4, "H": 32, "S": 4096, "D": 128}, "dtype": "bf16", "causal": True}
attn = attention_fwd.run(attn_cell)
```

```
caliper doctor            # is this machine set up to produce trustworthy numbers?
caliper fingerprint --check   # is the machine record complete?
caliper bench corpus:o1 --recording session.jsonl
caliper sweep spec.yaml --resume        # run a matrix -> results file
caliper validate results.parquet        # check a results file (or a bundle dir)
caliper compare --baseline base.parquet --candidate new.parquet --fail-on-regression
caliper submit results.parquet --out bundle/   # pack a bundle for caliper-results
caliper selftest --full                 # run the on-device oracle suite
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

Optional extras: `caliper-gpu[sweep]` (PyYAML, for reading sweep specs),
`caliper-gpu[parquet]` (pyarrow, for `Grid.to_parquet` / `caliper validate` on
`.parquet`), and `caliper-gpu[triton]` (torch + triton, for the reference
kernel corpus -- installs anywhere, only needs a CUDA device to actually run).
The library targets Linux with an NVIDIA GPU (CUDA 12.1+); the core and its
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
the Rust suite and the `l0`/`l1` Python tests on every push; the GPU tiers run
on Colab. [`CONTRIBUTING.md`](CONTRIBUTING.md) has the push -> Colab -> PR loop,
plus how to add an architecture, a corpus kernel, or a device backend.

## Submit your GPU's numbers

The roofline needs real per-SKU data, and the corpus needs runs on more than one
card. If you have a GPU:

```bash
caliper selftest --full --json > selftest.json     # confirm the box is fit
caliper sweep examples/corpus.yaml --parquet rows.parquet   # or your own kernels
caliper submit rows.parquet --out bundle/ --calibration <measured_us> <expected_us>
caliper validate bundle/                            # must print OK
```

Then open a PR adding `bundle/` under `results/` in
[`caliper-results`](results-repo/) -- the same `caliper validate` runs in the PR
gate. Full instructions: [`results-repo/SUBMITTING.md`](results-repo/SUBMITTING.md).

For a Tier-1 acceptance pass, run [`notebooks/acceptance.ipynb`](notebooks/acceptance.ipynb)
(the scriptable half of [`docs/acceptance/manual-playbook.md`](docs/acceptance/manual-playbook.md))
and file the report under `docs/acceptance/reports/`.
[`docs/acceptance/traceability.md`](docs/acceptance/traceability.md) maps every
FR / NFR to its CI evidence and its outstanding on-device step.

## License

MIT -- see [LICENSE](LICENSE).
