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
  (baseline), `prime-odd` (remainder/tail paths), `llm-7b` / `llm-70b` (the
  three GEMMs in a Llama-2 decoder layer at prefill lengths 512 / 2048).
  `docs/shapes.md` records every number's source. Exposed as
  `caliper._core.{resolve_shape_library, shape_library_names}`.
- `caliper_core::spec` + `caliper._spec` -- parse a `sweep` spec (Appendix D
  YAML; Python reads the YAML, Rust validates and expands) into the
  deduplicated cartesian product of dtypes x layouts x resolved shapes, with a
  typed `SpecError` for every malformed field. `Cell::key` is the identity for
  dedupe and `--resume`; `spec::pending` drops finished cells. A golden
  `appendix_d.yaml` -> `appendix_d.cells.json` pins the expansion. New `pyyaml`
  runtime dependency.
- `caliper.Grid` -- a table of `Result` rows: `.to_json()` / `.from_json()`
  (nested shape), `.to_parquet()` / `.from_parquet()` (one flattened row per
  measurement + a derived `toolchain_hash` column, per Appendix C; needs the
  `caliper[parquet]` extra), and `.filter()` into a smaller `Grid`.
- `caliper validate <file>` -- check every record in a `.json` / `.jsonl` /
  `.parquet` results file against the schema; exit 1 on any invalid row.
  `caliper.validate_records()` returns the report.
