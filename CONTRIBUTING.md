# Contributing to caliper

## Setup

```bash
git clone https://github.com/mansoor-mamnoon/caliper && cd caliper
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # builds the Rust extension via maturin
```

Rust toolchain (stable) is required for the extension; `rustup` installs it.

## The two test tiers you can run anywhere

| Command | What it covers |
|---|---|
| `cargo test --all --all-features` | the pure Rust core (`caliper-core`), the fixture/replay device layer (`caliper-gpu`) |
| `pytest -m "l0 or l1"` | the same logic through the Python bindings, plus the CLI |
| `cargo fmt --all -- --check` · `cargo clippy --all-targets --all-features -- -D warnings` | Rust style + lints |
| `ruff check .` · `ruff format --check .` · `mypy --strict python tests` | Python style + types |

CI (`.github/workflows/ci-cpu.yml`) runs all of the above on every push. Keep it
green.

## GPU work: the push → Colab → PR loop

Anything on-device (`l2` / `l4` / `l6` markers, the real launcher, the oracle
suite, `nsys` / `ncu` cross-checks) runs on Colab, not in CI. The loop:

1. **Push** your branch.
2. Open **`notebooks/dev.ipynb`** in Colab on a GPU runtime (A100 preferred; T4
   works). It bootstraps Rust + `nsys`, pulls your branch, then runs, in order:
   - `cargo test --all --all-features` and `pytest -m "l0 or l1"` (a smoke
     re-run of what CI already checked),
   - `pytest -m "l2 or l4 or l6"` (on-device oracles, unlocked reproducibility,
     end-to-end) and the `--features cuda` Rust tests,
   - `caliper doctor`, `caliper fingerprint --check`, and
     `caliper selftest --full`.
3. **Paste the pass/fail tail** of each cell into the PR description.
4. For an acceptance milestone, also run **`notebooks/selftest.ipynb`** and
   **commit `selftest-report.json`**. It must `validate` (the notebook checks
   this). Once the on-device oracle runner lands, an A100 without `nsys` reads
   `PASS` / `coverage: reduced` with
   `not_validated: [clock_lock, ncu_crosscheck, powercap_throttle]`; until then a
   device-present run is `ERROR` (every oracle skipped) and the notebook cell
   fails — that is expected.

`caliper selftest` exit codes: `0` PASS, `1` FAIL, `2` ERROR (including "no
device", and — for now — "device present, oracle runner not yet wired").

## Conventions

- Internals are Rust in `crates/caliper-core`; Python is the public API, the
  CLI, and bindings glue; CUDA C++ is only for the `.cu` oracle kernels.
- `caliper-core` forbids `unsafe` and depends only on `serde` / `serde_json`.
- Every result-affecting change gets a `CHANGELOG.md` entry under *Unreleased*,
  feature-scoped.
- New device-layer behaviour needs a fixture under
  `crates/caliper-gpu/fixtures/` and an L1 test; regenerate fixtures with the
  `_generate.py` next to them (they must be reproducible — `git diff` clean).
- The public API and CLI docs (`docs/api.md`, `docs/cli.md`) are checked against
  the code by `tests/l0_unit/test_docs_match_code.py`; update them in the same
  PR as a signature or subcommand change.

## Extending caliper

### Add an architecture to the roofline / occupancy models

`crates/caliper-core/src/roofline.rs` holds the per-`(arch, dtype)` peaks table;
`crates/caliper-core/src/occupancy.rs` holds the SM resource limits. Every new
cell needs a `// source:` comment citing an NVIDIA/AMD whitepaper, a
microbenchmarking paper, or a measured `caliper selftest` value (Appendix G).
Add the arch to the parametrised `cargo test` lists and to
`crates/caliper-core/tests/occupancy/reference.csv`. No Python change is
needed — the tables are pure Rust.

### Add a corpus kernel

Put it in `python/caliper/corpus/kernels/<name>.py` alongside `gemm.py`: a
guarded `try: import triton` block, a `KERNEL_KEY` that matches an entry in
`crates/caliper-gpu/src/corpus.rs::REFERENCE_TARGETS`, a `SOURCE_HASH =
content_hash(__file__)`, a pure `roofline_spec(shape, dtype)` that delegates to
`caliper.corpus._common.roofline_spec_for`, and a `run(cell, config=None) ->
Result`. Add the FLOP / HBM-byte formula as a new arm in
`roofline::corpus_spec` (Rust, `cargo test`), and an L0 test in
`tests/l0_unit/test_corpus_kernels.py` (import-without-triton, roofline math,
`NotImplementedError` off-GPU). See [`docs/corpus.md`](corpus.md).

### Add a device backend

The device layer is four ports (launch, clocks, device info, module probe) in
`crates/caliper-gpu`. A new backend (ROCm, a remote agent, ...) implements those
traits, plus a `record`/`fixture` capture so its behaviour is testable with no
hardware. Wire it into `open_from_env()` behind a `CALIPER_GPU_PORTS` value and a
Cargo feature; keep the `real` implementations feature-gated so the pure-core
build still compiles anywhere.

### Run the acceptance playbook for a GPU you have

Follow [`docs/acceptance/manual-playbook.md`](acceptance/manual-playbook.md)
(Playbook A on Colab, Playbook B on a root instance). File the filled report
under `docs/acceptance/reports/` and, for a corpus sweep, `caliper submit` the
rows to `caliper-results`.
