# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from the first
tagged release onward.

## [Unreleased]

### Added
- Project scaffold: a Cargo workspace with `caliper-core` (pure Rust measurement
  logic) and `caliper-ffi` (PyO3 bindings), built into a Python package with
  maturin. Ruff + Mypy for Python, rustfmt + Clippy for Rust, pytest markers
  `l0`/`l1`/`l2`/`l3`/`l4`/`l6`.
- `caliper_core::schema`: the versioned `Record` type and its sections
  (`KernelLabel`, `Timing`, `Roofline`, `Ptxas`, `Occupancy`, `Clocks`,
  `Machine`, `Toolkit`), with canonical JSON serialisation, lenient parsing
  (unknown keys ignored, missing sections defaulted), and a `validate` pass for
  hardware-independent invariants.
- `caliper._core` extension module exposing `schema_version`, `core_version`,
  `default_record_json`, `normalize_record_json`, and `validate_record_json`.
- Python `caliper.Result`: a dict-backed handle over the Rust schema, plus
  `caliper.schema_version()` and the `caliper` CLI entry point (`--version`,
  `--help`).
- Continuous integration: a Rust lane (`cargo fmt --check`, `clippy -D warnings`,
  `cargo test` with and without `--all-features`) and a Python lane (Ruff, Mypy,
  `l0`/`l1` tests, wheel build) across Python 3.10-3.12.
- `caliper_core::stats` -- p10/p50/p90 (NumPy-linear interpolation), raw MAD,
  sample coefficient of variation, and a cross-pass CoV; rejects empty or
  non-finite input.
- `caliper_core::warmup` -- `steady_state()`, which finds the first warm sample
  by walking a trailing-window median down to within a relative tolerance of the
  series' settled value. Exposed alongside `stats` through `caliper._core`
  (`summarize`, `cross_pass_cov`, `steady_state_index`).
- `caliper-gpu` -- the device layer. Three ports (`KernelLauncher`, `GpuClock`,
  `DeviceInfo`), a `FixturePlayer` that replays a recorded JSON Lines session as
  any port with no GPU, a `Recorder` that wraps a real port and writes that
  recording, and seed fixtures. The on-device implementations in `real` (behind
  the `cuda` feature) are typed stubs for now; they are filled in and validated
  on a CUDA host.
- `caliper_core::pipeline` -- `reduce()`, the pure heart of `bench()`: invalidate
  throttled batches, trim to steady state, convert per-batch to per-launch,
  summarise, and set advisory flags (`clocks-unlocked`,
  `throttled-samples-dropped`, `l2-flush-disabled`, `warmup-not-converged`).
  Plus `invalidate()` and `flush_buffer_bytes()` (L2-flush buffer sized from the
  device's L2, not a fixed 256 MiB constant).
- `caliper_gpu::bench` -- `run()` drives a device layer through one measurement
  (snapshot -> lock -> throttle-poll -> time -> throttle-poll -> read -> unlock
  -> reduce; per-batch "during" polling is the launcher's job);
  `run_replay()` does it from a recorded session, and requires the recording to
  be fully consumed. A clock-lock refusal (including a hard `PermissionDenied`)
  degrades to an unlocked, `clocks-unlocked`-tagged run rather than raising.
  Exposed to Python as `caliper.bench(recording=...)`, which returns a populated
  `Result`; passing a live kernel is not supported yet (needs the on-device
  launcher).
- Warm-up handling supports a fixed trim: `WarmupPlan { fixed: Some(n) }` /
  `caliper.bench(warmup=25)` skips detection and drops exactly `n` leading
  samples; `warmup="auto"` keeps steady-state detection.
- `GraphMode` (`auto` / `on` / `off`) replaces the `cuda_graph` bool in
  `BenchOpts` and `caliper.bench`; `auto` is resolved on-device.
- `CALIPER_GPU_PORTS` (`real` / `fixture` / `record`) + `CALIPER_GPU_FIXTURE`
  select the device backend at runtime via `caliper_gpu::open_from_env()` ->
  `DeviceLayerHandle` (a concrete enum implementing every port).
- `caliper.Result` gains the frozen read surface: `.p50_us` / `.p10_us` /
  `.p90_us` / `.mad_us` / `.wall_p50_us` / `.launch_overhead_us` /
  `.achieved_tflops` / `.flags` / `.throttle_reasons` / `.timing` / `.machine` /
  `.ptxas` / etc.
- Fixture recordings may carry `#`-prefixed header/comment lines
  (`# caliper-fixture v=... arch=...`).
- `caliper_core::oracles` -- analytic expectations and pass/fail checks for the
  self-check kernels O1 (calibrated duration), O2 (streaming triad / L2 flush),
  O3 (FMA peak), O4 (launch overhead), O6 (throttle detection), plus a
  least-squares `fit_line`. The kernels themselves are `caliper-gpu/kernels/
  oracles.cu` (compiled and run on a CUDA host).
- `caliper_core::doctor` -- `assess()` turns gathered device facts into a
  verdict (`fit` / `unfit` / `error`), an `environment` (`normal` /
  `constrained`), per-check detail, notes, and an exit code. `caliper-gpu` adds
  `doctor::gather` / `doctor::run` over a device layer.
- CLI: real `caliper bench` (recorded-session path, `--json`, `--warmup`,
  `--cuda-graph`, `--no-flush-l2`, `--no-lock-clocks`), `caliper doctor`
  (exit 0 fit / 1 unfit / 2 error; the non-`--json` output follows the
  honest-degradation wording of the plan), and `caliper fingerprint`.
  `caliper bench corpus:o1..o4, o6` resolves the target to the built-in oracle
  kernel key; an unknown `corpus:*` target is rejected.
- `caliper.doctor()` / `caliper.fingerprint()` Python entry points.
- `notebooks/dev.ipynb` (the Colab "GPU CI") and `make sync`.
- `caliper_core::ptxas_parse` -- one output shape (`ParsedKernel`, mapping onto
  `Record.ptxas`) from three compiler reports: `ptxas -v` (single and
  multi-kernel, spills, shared memory), `cuobjdump -res-usage`, and the HIP /
  `amdgpu` `; NumVgprs:` comment block. Malformed / empty / unrecognised input
  -> `PtxasParseError`. Exposed as `caliper._core.parse_ptxas`. `ptxas` cannot
  know a kernel's dynamic shared memory, so `smem_dynamic_bytes` stays `None`.
- `caliper-gpu` gains a `ModuleProbe` port (fixture + `Recorder` + feature-gated
  `real` stub), now part of the `DeviceLayer` bound. `bench::run` probes the
  compiled module after timing and fills `Record.ptxas`; a probe that is not
  available flags the record `ptxas-unavailable` rather than failing the run.
- `caliper_core::occupancy` -- the CUDA theoretical-occupancy model: from an
  architecture, registers per thread, shared memory per block, and block size it
  computes resident blocks per SM, active warps, the occupancy fraction, and
  which resource is the limiter, plus a scheduling-wave count. Verified against a
  checked-in CUDA Occupancy Calculator reference table
  (`crates/caliper-core/tests/occupancy/reference.csv`) covering Volta through
  Blackwell. Exposed as `caliper._core.theoretical_occupancy`.
- `caliper_core::roofline` -- a per-architecture, dtype-aware peaks table (FP32
  FMA, tensor-core dense for fp16/bf16/tf32/fp8, and HBM bandwidth for SM70/75/
  80/86/89/90/120 and CDNA3), every cell carrying a `source:` citation to a
  vendor whitepaper or datasheet. `analyze()` reports achieved TFLOP/s and GB/s
  (matching the O2/O3 oracle formulas), arithmetic intensity, the ridge point,
  and a `bound` of `compute` / `memory` / `latency` / `unknown`. Exposed as
  `caliper._core.roofline_analyze`, `peak_compute_tflops`, `peak_hbm_gbps`.
- `reduce()` now fills `Record.occupancy` and `Record.roofline` when the caller
  supplies launch geometry (`block_size`, `grid_blocks`) and a roofline spec
  (dtype + FLOP / HBM-byte counts) on `ReduceInput`; both sections stay empty
  otherwise.
- `ModuleProbe` gains `max_active_blocks_per_sm` -- the driver's
  `cuOccupancyMaxActiveBlocksPerMultiprocessor` for a launch. The default and
  the fixture player answer "unavailable"; the `real` CUDA probe returns a
  pending stub until the driver call is wired on a CUDA host. When it and the
  occupancy model disagree by more than one block the record is flagged
  `occupancy-model-mismatch`. `RawSamples` carries `block_size` / `grid_blocks`
  / `dynamic_smem_bytes` so the launcher can feed both the model and the check.
- `caliper_core::fingerprint` -- a completeness check for the machine block:
  every hardware / driver / `nvcc` / `ptxas` field is *required* (a gap makes
  the fingerprint incomplete and `caliper fingerprint --check` exit 1), Triton
  and PyTorch versions are *recommended*. Plus `parse_nvcc_version` /
  `parse_ptxas_version` for the `Cuda compilation tools, release X, V X.Y.Z`
  line. Exposed as `caliper._core.fingerprint_check` /
  `fingerprint_is_complete` / `parse_{nvcc,ptxas}_version` and
  `caliper.fingerprint_check()`.
- `caliper._toolchain` / `caliper.toolchain()` -- detect the local Triton,
  PyTorch, `nvcc`, and `ptxas` versions (package metadata + `--version`
  subprocesses parsed by the Rust core), each `None` when absent.
- `caliper fingerprint --check` reports fingerprint completeness and exits 1 on
  a missing required field.
- `docs/checklist_fingerprint.md` -- every machine field mapped to its
  `nvidia-smi -q` / NVML / `--version` source, with a Colab dry-run.
- `caliper.do_bench` -- a Triton-compatible `do_bench` shim (`quantiles`,
  `return_mode`, `grad_to_none` / `fast_flush` / `warmup` / `rep` accepted for
  parity) that returns milliseconds. A live callable is timed with CUDA events
  the way Triton's own `do_bench` does (needs PyTorch + CUDA); a `recording=` /
  `fixture=` replays a recorded session with no GPU. Backed by
  `bench_replay_quantiles`, `caliper._core.quantiles`, and the new
  `timing.mean_us` / `min_us` / `max_us` schema fields; `Result` gains
  `.mean_us` / `.min_us` / `.max_us`.
- `bench(corpus:*)` fills in `Record.roofline`: `roofline::corpus_spec` infers
  the FLOP / HBM-byte counts for `corpus:gemm` (from `shape={"M","N","K"}` +
  `dtype`), `oracle:triad`, and `oracle:fma_peak`. `bench()` gains a `shape=`
  argument; `BenchOpts` a `roofline` field. `corpus:gemm` is added as a
  reference target.
- `caliper_core::graph` -- the `cuda_graph="auto"` capture policy:
  `should_capture` / `resolve` against a single-launch threshold
  (`DEFAULT_SINGLE_LAUNCH_THRESHOLD_US`). `bench` records the outcome as a
  `graph-captured` / `graph-eager` flag, from the launcher's own report
  (`RawSamples.graph_used` / `single_launch_us`) or the policy.
- `caliper_core::selftest` + `caliper selftest [--full] [--json]` -- the
  Appendix-E oracle self-test report: `SelftestReport::assemble` scores the
  O1-O7 / reproducibility / (`--full`) O5 + `nsys` checks (context lines like
  `device_present` do not count) into a `PASS` / `FAIL` / `ERROR` -- a run where
  no scored check passed is an `ERROR`, a `reduced`-coverage run with every
  runnable check passing is still a `PASS`. `coverage` is `full` only when the
  `nsys` cross-check ran; `not_validated` is the constrained-host capability set
  (`clock_lock` / `ncu_crosscheck` / `powercap_throttle`). `validate()`
  re-derives the result / coverage, rejects an unknown `not_validated` token,
  and flags a fabricated `PASS`. Exit codes 0 / 1 / 2; with no device (and,
  until the on-device oracle runner lands, with a device but every oracle
  skipped) the report is `ERROR`. Exposed as `caliper.selftest()` /
  `SELFTEST_EXIT_CODE` and `caliper._core.{selftest_from_env, selftest_assemble,
  validate_selftest_json}`.
- `caliper_core::oracles` gains O7: `check_o7_calibration_gemm(measured, arch)`
  compares a locked-clock calibration-GEMM `p50` against `calibration_gemm_p50_us`
  within +-8% (`verified` / `clocks-suspect`). It returns `None` for a SKU not
  in the table (the caller reports `SKIP`); the table ships empty until each
  SKU is measured at acceptance.
- `notebooks/selftest.ipynb` (runs `caliper selftest --full`, saves and
  validates the report) and a finalized `notebooks/dev.ipynb`; `CONTRIBUTING.md`
  documents the push -> Colab -> PR loop.
- `caliper_core::shapes` -- named shape libraries for `sweep`: `square-pow2`
  (baseline), `prime-odd` (remainder/tail paths), `llm-7b` (3 GEMMs/layer,
  MHA) and `llm-70b` (4 GEMMs/layer -- grouped-query attention gives a distinct
  smaller K/V projection), at prefill lengths 512 / 2048. `docs/shapes.md`
  records every number's source. Exposed as
  `caliper._core.{resolve_shape_library, shape_library_names}`.
- `caliper_core::spec` + `caliper._spec` -- parse a `sweep` spec (Appendix D
  YAML; Python reads the YAML, Rust validates and expands) into the
  deduplicated cartesian product of dtypes x layouts x resolved shapes, with a
  typed `SpecError` for every malformed field (unknown key, empty / unknown
  dtype / layout, bad `warmup` / `cuda_graph` / `autotune`, zero `min_samples`,
  unknown shape library, ...). Inline shapes accept a bare `{M, N, K}` /
  `{B, H, S, D}` (any case) as well as the tagged form. `Cell::key` is the
  identity for dedupe and `--resume`; `spec::pending` drops finished cells. A
  golden `appendix_d.yaml` / `appendix_d.cells.json` pair pins the expansion
  from both the Python (YAML) and Rust (JSON) sides. PyYAML is an optional
  dependency (`caliper[sweep]`).
- `caliper.Grid` -- a table of `Result` rows: `.to_json()` / `.from_json()`
  (nested shape), `.to_parquet()` / `.from_parquet()` (one flattened row per
  measurement + a `toolchain_hash` column -- sha256 of the sorted toolkit map
  and driver, per Appendix C; needs the `caliper[parquet]` extra), and
  `.filter()` into a smaller `Grid`.
- `caliper validate <file>` -- check every record in a `.json` / `.jsonl` /
  `.parquet` results file against the schema; exit 1 on any schema-invalid or
  wrong-typed row, exit 2 on an unreadable file. `caliper.validate_records()`
  returns the report.
- `caliper_core::autotune::AutotuneKey` + `caliper._autotune.AutotuneCache` --
  a per-autotune-config timing cache keyed by `(sku, driver, cuda, ptxas,
  triton, torch, kernel_source_hash, canonical-config-JSON)`. Adding a config
  invalidates only that config's key, so a re-sweep re-times just the new one.
  The JSON-file store has hit/miss counters and an atomic flush.
- `caliper.sweep(spec)` + `caliper sweep <spec.yaml>` -- expand a spec, run each
  cell (one `bench()` call per autotune config via `configs_for`, the fastest
  kept), and return a `Grid`. Each cell is checkpointed to a
  `<output>.state.jsonl` sidecar, so `--resume` (or the spec's `output.resume`)
  continues a killed run without re-measuring finished cells and keeps spec cell
  order; outputs are written once at the end via a temp file + atomic rename.
  `run_cell=` overrides how a `(cell, config)` is measured; the default replays
  a `<cell-key>.jsonl` recording from `recordings_dir`. `cache_path=` wires the
  `AutotuneCache` in, so a re-sweep re-times only a newly-added config
  (`tests/l6_e2e/test_autotune_cache.py` on a Colab A100; the hit/miss logic is
  covered on the no-GPU path). `bench()` gains a `layout=` argument, threaded
  onto the kernel label.
- `caliper.live_timing_ms(fn, warmup=25, rep=100, grad_to_none=None)` -- the
  CUDA-event timing loop factored out of `do_bench()` (à la
  `triton.testing.do_bench`), returned as raw unreduced per-launch millisecond
  samples; `do_bench()` now just reduces them. `_check_live_deps(caller)` is
  the shared "needs PyTorch + a CUDA device" guard both paths raise through.
- The reference kernel corpus (`python/caliper/corpus`): `gemm`, `rmsnorm`,
  `softmax`, each a Triton implementation plus a vendor baseline
  (`torch.matmul` / a plain-torch reference / `torch.softmax`). Unlike the
  CUDA-C++ oracle kernels, these don't need the (still-stubbed) Rust launcher
  -- `run()` times itself directly via `live_timing_ms` and builds its own
  machine fingerprint from `torch.cuda` device introspection, so the corpus
  genuinely runs end to end on any CUDA host today. Every kernel module
  imports cleanly without Triton installed; only `run()` needs it. See
  `docs/corpus.md` for the roofline formula, baseline, and Triton-API pin
  behind each one. `gemm.kernel` is `@triton.autotune`-wrapped over
  `gemm.CONFIGS` (5 block tilings), satisfying the autotune-cache contract in
  `tests/l6_e2e/test_autotune_cache.py`. `caliper-gpu[triton]` (torch +
  triton) is a new optional extra.
- `roofline::corpus_spec` gains `rmsnorm` and `softmax` arms (needs `ROWS` /
  `COLS` in `shape`): 4 and 5 FLOPs/element respectively, `2*ROWS*COLS*
  dtype_bytes` HBM traffic for both (read the input, write the output).
  `corpus:rmsnorm` and `corpus:softmax` are added as reference targets
  alongside `corpus:gemm`.
- `corpus.kernels.attention_fwd` / `attention_bwd` -- FlashAttention-style
  forward and backward Triton kernels plus their
  `F.scaled_dot_product_attention` baselines. All five FR-14 kernels now have
  a Triton implementation and a vendor baseline; on-device acceptance
  (valid rows on SM80/86/89/90) and the attention fp8 path for L4 are still
  pending on Colab. Forward: online-softmax running max/sum, causal mask,
  grouped-query attention (`h_kv` < `h`), head dim 64/128, bf16/fp16/fp32
  (fp8 raises `NotImplementedError`). Backward: a `delta = rowsum(dO*O)`
  preprocess kernel, then one K/V block per program accumulating `dK`/`dV`
  with `dQ` via atomics into an fp32 scratch; GQA by expand-then-group-reduce.
  Both sides are timed backward-only (the baseline via `autograd.grad` over
  one untimed forward). Each module exposes `check_numerics(cell)` returning
  an `allclose` verdict against the baseline (the DoD's correctness gate).
  `roofline::corpus_spec`
  gains an `attention` arm (needs `B`/`H`/`S`/`D`, optional `causal`):
  `4*B*H*S*S*D` FLOPs for the two matmuls (softmax omitted, <2%), halved for
  `causal`, `2.5x` for the backward; IO-aware `bytes_hbm` of `4*` (fwd) or
  `8*` (bwd) `B*H*S*D*dtype_bytes`. `corpus:attention_fwd` /
  `corpus:attention_bwd` added as reference targets. `assemble_result()` now
  takes the machine fingerprint as an argument (pure, off-device testable).
- `caliper_core::thresholds` + `caliper.compare()` + `caliper compare` -- the
  variance-aware regression diff. Two datasets are aligned by facet (kernel,
  impl, dtype, shape, layout, arch); each facet's candidate median is judged
  against a noise band derived from the baseline's MAD (`MAD -> sigma` via
  1.4826, then `sigma_mult` sigmas, with a 2% relative floor and a 50% cap so
  neither a suspiciously tight nor a suspiciously noisy baseline can mislead)
  or an explicit `--threshold PCT`. Each facet reports `delta` and `band` as
  fractions, the per-field `ptxas` and occupancy deltas (both shown in the
  human output for a regressed / improved / spilling facet), a
  `spill_regression` flag when the candidate spills more, and the autotune
  configs a facet had in the baseline but lost. `any_regression` (what
  `--fail-on-regression` keys off) covers a timing *or* a spill regression --
  a spill regression fails the run even when `--threshold` would forgive the
  slowdown. `caliper compare --baseline --candidate [--arch] [--threshold]
  [--fail-on-regression] [--json]` exits 0 ok / 1 regression / 2 error, and
  warns on stderr when `--arch` matches no rows; `_load_records` (shared with
  `validate`) reads `.json` / `.jsonl` / `.parquet`. Exposed as
  `caliper._core.compare_datasets`. New fixtures
  `tests/testdata/{base,slow,spill}.{json,parquet}` (+ `build_parquet.py`).
- `caliper_core::submit` + `caliper.submit()` + `caliper submit` -- the
  results-bundle flow. A bundle is `manifest.json` + `rows.parquet` +
  `fingerprint.json`; the manifest summarises the arch, the Appendix-C
  toolchain hash, the clock-lock tier, the sorted kernel names, and -- when the
  rows contain them -- a determinism-repeat CoV (vs the 2%-locked / 5%-unlocked
  tolerance) and a calibration-GEMM ratio (vs +/-8%). `caliper submit FILE...
  [--out DIR] [--repo DIR] [--dry-run] [--calibration MEASURED EXPECTED]`
  builds it; with `--repo` a local `caliper-results` checkout and
  `--dry-run=false` the bundle lands on a fresh branch there.
- `caliper validate` is now the shared results gate. A directory is read as a
  bundle and checked by `caliper_core::submit::validate_bundle` -- per-row
  schema validity, the submission-strict extras (required fields,
  `roofline_pct <= 1.05` rather than the schema's 1.5 recording clamp), the
  manifest / rows / fingerprint arch consistency, a within-bundle
  exact-duplicate row check, and the determinism (NFR-5 duration-banded CoV:
  2% locked, 5% unlocked >=100us, 8% unlocked 10-100us) and calibration
  (+/-8%) verdicts. The tier, kernel list, and those verdicts are
  **recomputed from the bundled rows** -- a submitter-controlled manifest's
  own `within_tolerance` booleans are not trusted. Cross-bundle dedupe (a PR
  whose facet already exists under `results/`) is deferred to the
  `caliper-results` CI. Exposed as `caliper._core.submit_manifest` /
  `validate_bundle`.
- `results-repo/` -- the `caliper-results` scaffold: the
  `results/<arch>/<toolchain-hash>/` layout, `SUBMITTING.md`, a `schema/`
  pointer (the validator is the schema), and a PR workflow that runs `caliper
  validate` on each changed bundle. Fixtures
  `tests/testdata/{bundle_ok,bundle_missing_field,bundle_nonreproducing,bundle_slow_calibration}/`
  + `over_peak_row.json` (+ `build_bundles.py`) cover playbooks #13 and #14.
- `docs/why-do_bench-misleads.md` -- the launch writeup ("Your Triton benchmark
  is probably lying to you"): four ways a default `do_bench` call drops context
  that changes the answer (per-launch event tax on short kernels, a fixed
  warmup that misses the clock ramp, an L2-resident working set with no flush,
  unlocked/unrecorded clocks), each with the mechanism, the upstream issue
  links, and what `caliper` does instead. `examples/misleads/` has the three
  runnable experiments (`fast_kernel`, `cold_warmup`, `l2_resident`);
  `make writeup-data` runs them into `docs/data/misleads.csv`.
- Reference docs: `docs/api.md` (every public `caliper` symbol + the record /
  Parquet-row schema), `docs/cli.md` (every subcommand, its flags, and the exit
  codes), and `docs/acceptance/manual-playbook.md` (the 14-step L5 playbook,
  split into Playbook A / B, with the report template).
  `tests/l0_unit/test_docs_match_code.py` fails if a public symbol, a schema
  field, or a CLI subcommand is undocumented. `CONTRIBUTING.md` gains an
  "Extending caliper" section (add an architecture / a corpus kernel / a device
  backend / run the playbook).
- `notebooks/quickstart.ipynb` + an "Open in Colab" badge: install from source,
  `caliper doctor`, a replay `caliper bench`, and a pure `roofline_spec` -- a
  new reader runs it in a fresh Colab runtime with no local setup. The README
  gains a badge row, a "30 seconds" block, and a "Submit your GPU's numbers"
  section.
- Acceptance harness: `notebooks/acceptance.ipynb` works through the
  scriptable Playbook-A steps on a Colab GPU -- version / doctor / selftest,
  the `l2`/`l4`/`l6` tiers, a 20-cell `corpus:gemm` sweep
  (`examples/acceptance-sweep.yaml` = `square-pow2` x 2 dtypes x 2 layouts) ->
  `validate` -> `submit --dry-run`, the compare + negative-validate fixtures,
  and the `do_bench` shim delta -- then writes a filled `report.md` and
  `selftest-<arch>.json` to commit under `docs/acceptance/`.
  `docs/acceptance/traceability.md` maps every FR-1..FR-19 / NFR-1..NFR-10 to
  its CI evidence and its outstanding on-device step;
  `tests/l0_unit/test_acceptance_traceability.py` fails if a requirement is
  unmapped or the notebook / sweep spec drifts.
- Release path: `.github/workflows/release.yml` fires on a `vX.Y.Z` tag --
  builds the sdist and cp310-cp312 manylinux wheels, checks the tag against the
  crate version, publishes to Test PyPI and then (behind a manual approval on
  the `pypi` environment) to PyPI via OIDC trusted publishing, and opens a
  GitHub Release whose notes come from the matching `CHANGELOG.md` section with
  the wheels, every `selftest-*.json`, the golden-box `ncu` report, and the
  do_bench writeup attached. `workflow_dispatch` runs the build + Test PyPI
  rehearsal only. `RELEASING.md` is the checklist: the `docs/plan.md` §5
  Definition-of-Done gate, the version bump, the tag, and the post-release
  fresh-Colab `pip install` check.
- Acceptance triage + Tier-2: `docs/acceptance/triage.md` is the deviation loop
  (file -> classify Tier-1 blocking / Tier-2 best-effort -> fix -> re-run the
  affected step -> close) with a log table and an
  `acceptance-deviation` issue template; `docs/acceptance/tier2.md` sets the
  reduced bar for SM86 (full Playbook A) and MI300X / CDNA3 (`doctor` + one
  corpus kernel + `validate` only), both explicitly not release-blocking.
