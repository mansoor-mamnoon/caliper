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
  (snapshot -> lock -> time -> read -> unlock -> reduce); `run_replay()` does it
  from a recorded session. Exposed to Python as `caliper.bench(recording=...)`,
  which returns a populated `Result`; passing a live kernel is not supported yet
  (needs the on-device launcher).
