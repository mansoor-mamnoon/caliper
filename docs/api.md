# Python API

Everything importable from `caliper`. Signatures are the source of truth in
`python/caliper/`; `tests/l0_unit/test_docs_match_code.py` checks this page
lists every public symbol.

```python
from caliper import bench, do_bench, sweep, compare, submit, Result, Grid
```

## Measurement

### `bench(target=None, *, fixture=None, recording=None, kernel_key="kernel", kernel_impl=None, dtype=None, layout=None, shape=None, batch=32, batches=50, cuda_graph="auto", flush_l2=True, lock_clocks=True, sm_mhz=None, mem_mhz=None, warmup="auto", warmup_window=20, warmup_tol=0.02, warmup_min=30) -> Result`

The rigorous single-kernel measurement: snapshot the machine, lock clocks, poll
for throttling, time a batch of launches, reduce to a distribution, attach the
roofline / occupancy / ptxas context. Returns a [`Result`](#result).

Today only the **replay** path is wired: pass `recording=` the text of a JSON
Lines device session, or `fixture=` a path to one. A live `target` needs the
on-device launcher (a CUDA-host stub); it raises `NotImplementedError`.

- `warmup` -- `"auto"` for steady-state detection, or an `int` to trim exactly
  that many leading samples.
- `cuda_graph` -- `"auto"` / `"on"` / `"off"` (or a bool).
- `shape` -- e.g. `{"M": 4096, "N": 4096, "K": 4096}`; used to fill the roofline
  for a `corpus:*` target.

### `do_bench(fn=None, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean", *, fixture=None, recording=None, kernel_key="kernel", dtype=None, batch=32, batches=50) -> float | list[float]`

Drop-in for `triton.testing.do_bench` -- same argument order, result in
**milliseconds**. A live `fn` is timed with CUDA events (needs PyTorch + CUDA);
`fixture=` / `recording=` replays a session with no GPU. `return_mode` is
`"min"` / `"max"` / `"mean"` / `"median"`; `quantiles` (e.g. `[0.5, 0.2, 0.8]`)
overrides it and returns a list. For the clock-locked measurement use
[`bench`](#bench).

### `live_timing_ms(fn, warmup=25, rep=100, grad_to_none=None) -> list[float]`

The CUDA-event timing loop from `do_bench`, returned as the raw per-launch
millisecond samples (unreduced). Needs PyTorch + a CUDA device.

## Sweeps and regressions

### `sweep(spec, *, recordings_dir=None, run_cell=None, configs_for=None, cache_path=None, machine=None, kernel_source_hash=None, parquet=None, json_out=None, resume=None, state_path=None) -> Grid`

Expand a spec (a dict, a `Path`, a filename, or YAML text -- Appendix D), run
every cell, checkpoint after each to a `<output>.state.jsonl` sidecar, and
return a [`Grid`](#grid). `configs_for(cell)` returns the autotune configs to
time (default: one, no tuning); `run_cell(cell, config)` overrides how a cell is
measured (default: replay `<cell-key>.jsonl` from `recordings_dir`).
`cache_path=` wires an autotune-config timing cache so a re-sweep re-times only
a newly-added config. `resume=True` (or the spec's `output.resume`) skips cells
already in the state file. At least one of `parquet=` / `json_out=` (or the
spec's `output:` block) is required.

### `compare(baseline, candidate, *, arch=None, threshold=None, sigma_mult=3.0, floor_pct=0.02, fail_on_regression=False) -> dict`

Diff two results files (`.json` / `.jsonl` / `.parquet`) for variance-aware
performance regressions. Rows align by *facet* (kernel, impl, dtype, shape,
layout, arch). Each facet's candidate median is judged against a band derived
from the baseline's MAD (`sigma_mult` sigmas, `floor_pct` relative floor, capped
at 50%), or an explicit `threshold` (a fraction, e.g. `0.10` for 10%). The
report carries per-facet `delta` / `band` (fractions), the `ptxas` and occupancy
deltas, a `spill_regression` flag, and any dropped autotune configs.
`report["any_regression"]` is true for a timing *or* a spill regression -- a
spill regression fails the run even under an explicit `threshold`. With
`fail_on_regression` the report also carries `"exit_code"` (0 or 1).

## Submitting

### `submit(paths, *, out=None, repo=None, dry_run=True, calibration=None) -> dict`

Build a `caliper-results` submission bundle from one or more results files: a
`manifest.json` (arch, toolchain hash, clock-lock tier, kernels, and -- when the
rows contain them -- a determinism-repeat CoV and a calibration ratio) +
`rows.parquet` + `fingerprint.json`. With `out=` the three files are written
there. `calibration` is `(measured_p50_us, expected_p50_us)` for the SKU's
calibration GEMM. With `dry_run=False` and `repo=` a path to a local
`caliper-results` checkout, the bundle is committed to a fresh branch there
(push + PR are manual); a repo *URL* is refused.

### `validate_records(path) -> dict`

Validate a `.json` / `.jsonl` / `.parquet` results file (every record against
the schema) *or* a bundle directory (`manifest.json` + `rows.*` +
`fingerprint.json`, run through the shared bundle gate: schema + submission-strict
fields + `roofline_pct <= 1.05` + determinism / calibration / arch consistency).
Returns `{"n", "n_invalid", "problems", "ok"}` (bundles also carry `"bundle"`).

## Environment

### `doctor(*, fixture=None, recording=None) -> dict`  ·  `doctor_text(...) -> str`

Is this machine fit to produce trustworthy numbers? Returns the report dict (or
the terminal rendering). Sourced from the live device, or from a recorded
session.

### `fingerprint(*, fixture=None, recording=None) -> dict`  ·  `fingerprint_check(...) -> dict`

The machine fingerprint (GPU, driver, toolchain, ...), or a completeness report
(`{"complete", "missing_required", "missing_recommended"}`).

### `selftest(*, full=False) -> dict`

Run the Appendix-E oracle self-test against the `CALIPER_GPU_PORTS` backend.
Returns the report dict; map `report["result"]` through
`SELFTEST_EXIT_CODE = {"PASS": 0, "FAIL": 1, "ERROR": 2}`. `full=` also runs O5
(cuBLAS) and the `nsys` cross-check.

### `toolchain() -> dict[str, str | None]`

Detected local kernel toolchain (`triton`, `torch`, `nvcc`, `ptxas`); each value
is a version string or `None`.

## Types

### `Result`

A dict-backed handle over the Rust schema. Construct with `Result.default()`,
`Result.from_dict(d)`, or `Result.from_json(s)`.

- **Serialise**: `.to_dict()`, `.to_json()` (canonical, stable key order).
- **Validate**: `.validate() -> list[str]` (empty == well-formed).
- **Scalars**: `.p10_us` / `.p50_us` / `.p90_us` / `.mean_us` / `.min_us` /
  `.max_us` / `.mad_us` / `.wall_p50_us` / `.launch_overhead_us` /
  `.achieved_tflops` / `.roofline_pct` / `.measured_at` / `.schema_version` /
  `.caliper_version`.
- **Sections** (attribute-or-key dicts): `.kernel` / `.timing` / `.roofline` /
  `.ptxas` / `.occupancy` / `.clocks` / `.machine`.
- **Lists**: `.flags` / `.throttle_reasons`.

### `Grid`

An ordered table of `Result` rows -- what a sweep returns.

- `Grid(rows)` -- `rows` is a sequence of `Result` or dict; each is
  schema-normalised on construction.
- `len(grid)`, iteration, `grid[i]`, `.rows() -> list[Result]`.
- `.filter(predicate) -> Grid`.
- `.to_json(path=None, *, indent=None) -> str` / `Grid.from_json(source)`.
- `.to_parquet(path)` / `Grid.from_parquet(path)` (one flattened row per
  measurement + a `toolchain_hash` column; needs `caliper-gpu[parquet]`).
- `.to_table() -> pyarrow.Table`.

### `schema_version() -> str`  ·  `__version__: str`

## The record / row schema (v1)

Every `Result` is one record. Every field is optional; a partially populated
record is still valid. JSON (nested) is Appendix B; the flat Parquet row is
Appendix C -- `Result` with dotted names (`timing.p50_us`,
`machine.sm_arch`, ...), free-form maps (`kernel.shape`,
`kernel.autotune_config`) kept as one JSON string, plus a `toolchain_hash`
column.

| Section | Fields |
|---|---|
| *(top)* | `schema_version`, `caliper_version`, `measured_at`, `host_id_class` |
| `kernel` | `name`, `impl`, `source_hash`, `autotune_config`, `dtype`, `shape`, `layout` |
| `timing` | `p10_us`, `p50_us`, `p90_us`, `mean_us`, `min_us`, `max_us`, `mad_us`, `wall_p50_us`, `launch_overhead_us`, `n_samples`, `n_warmup_to_steady`, `invalidated_samples`, `cross_pass_cov` |
| `roofline` | `achieved_tflops`, `roofline_pct`, `achieved_gbps`, `arithmetic_intensity`, `ridge_point`, `bound`, `baseline_pct`, `baseline` |
| `ptxas` | `regs_per_thread`, `smem_static_bytes`, `smem_dynamic_bytes`, `spill_loads_bytes`, `spill_stores_bytes`, `local_bytes`, `stack_bytes` |
| `occupancy` | `theoretical`, `achieved`, `active_warps_per_sm`, `waves` |
| `clocks` | `sm_mhz`, `mem_mhz`, `locked`, `lock_method` |
| `machine` | `gpu_name`, `sm_arch`, `vram_mib`, `sm_count`, `l2_bytes`, `bar1_mib`, `driver`, `cuda_runtime`, `cuda_driver`, `nvml_version`, `ecc`, `mig`, `persistence_mode`, `pcie_gen`, `pcie_width`, `toolkit{triton,torch,ptxas,nvcc}` |
| *(top)* | `throttle_reasons: [str]`, `flags: [str]` (e.g. `clocks-unlocked`, `corpus-live-timing`, `throttled-samples-dropped`, `ptxas-unavailable`) |

## `caliper.corpus`

The reference kernels (Triton implementation + vendor baseline + roofline).
Importable without Triton; `run()` needs Triton + PyTorch + a CUDA device.

```python
from caliper.corpus.kernels import gemm, rmsnorm, softmax, attention_fwd, attention_bwd

gemm.run({"shape": {"m": 4096, "n": 4096, "k": 4096}, "dtype": "bf16", "layout": "row"})
gemm.roofline_spec({"M": 4096, "N": 4096, "K": 4096}, "bf16")   # pure, no GPU
attention_fwd.check_numerics({"shape": {"B": 4, "H": 32, "S": 4096, "D": 128}, "dtype": "bf16"})
```

Each module exposes `KERNEL_KEY`, `SOURCE_HASH`, `roofline_spec(shape, dtype)`,
`run(cell, config=None) -> Result`; the two `attention` modules add
`check_numerics(cell) -> {"max_abs_err", "max_rel_err", "allclose"}`. See
[`corpus.md`](corpus.md).
