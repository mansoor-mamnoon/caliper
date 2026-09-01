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
   - `pytest -m "l2 or l6"` and the `--features cuda` Rust tests,
   - `caliper doctor`, `caliper fingerprint --check`, and
     `caliper selftest --full`.
3. **Paste the pass/fail tail** of each cell into the PR description.
4. For an acceptance milestone, also run **`notebooks/selftest.ipynb`** and
   **commit `selftest-report.json`** — it must `validate` (the notebook checks
   this) and, on an A100 without `nsys`, read `PASS` / `coverage: reduced` with
   `vs_nsys` (and `o5_cublas_gemm`) in `not_validated`.

`caliper selftest` exit codes: `0` PASS, `1` FAIL, `2` ERROR (including "no
device").

## Conventions

- Internals are Rust in `crates/caliper-core`; Python is the public API, the
  CLI, and bindings glue; CUDA C++ is only for the `.cu` oracle kernels.
- `caliper-core` forbids `unsafe` and depends only on `serde` / `serde_json`.
- Every result-affecting change gets a `CHANGELOG.md` entry under *Unreleased*,
  feature-scoped.
- New device-layer behaviour needs a fixture under
  `crates/caliper-gpu/fixtures/` and an L1 test; regenerate fixtures with the
  `_generate.py` next to them (they must be reproducible — `git diff` clean).
