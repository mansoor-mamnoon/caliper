//! PyO3 bindings for `caliper-core`.
//!
//! This crate is a thin marshalling layer: it converts between Python values and
//! `caliper-core` calls, and does no logic of its own. It is built by maturin
//! into the `caliper._core` extension module.

use std::collections::HashMap;

use caliper_core::{schema, stats, warmup};
use caliper_gpu::fixture::FixturePlayer;
use caliper_gpu::{corpus, doctor as gpu_doctor, run_replay, BenchOpts, DeviceInfo};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// --- schema -----------------------------------------------------------------

/// The result schema version this build understands.
#[pyfunction]
fn schema_version() -> &'static str {
    schema::SCHEMA_VERSION
}

/// The `caliper-core` crate version.
#[pyfunction]
fn core_version() -> &'static str {
    schema::CALIPER_VERSION
}

/// A default (empty) record, as canonical JSON.
#[pyfunction]
fn default_record_json() -> String {
    schema::to_json(&schema::Record::default())
}

/// Parse a record document and return it in canonical form. Raises
/// ``ValueError`` if the input is not valid JSON for a record.
#[pyfunction]
fn normalize_record_json(text: &str) -> PyResult<String> {
    schema::normalize_json(text).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Validate a record document. Returns a list of human-readable problems (empty
/// if the record is well-formed). Raises ``ValueError`` on a JSON parse error.
#[pyfunction]
fn validate_record_json(text: &str) -> PyResult<Vec<String>> {
    schema::validate_json(text).map_err(|e| PyValueError::new_err(e.to_string()))
}

// --- statistics -----------------------------------------------------------

/// Summarise timing samples into `{n, min, p10, p50, p90, max, mean, mad, cov}`.
/// Raises ``ValueError`` if the input is empty or contains a non-finite value.
#[pyfunction]
fn summarize(samples: Vec<f64>) -> PyResult<HashMap<String, f64>> {
    let s = stats::summarize(&samples)
        .ok_or_else(|| PyValueError::new_err("samples must be non-empty and finite"))?;
    let mut out = HashMap::new();
    out.insert("n".to_string(), s.n as f64);
    out.insert("min".to_string(), s.min);
    out.insert("p10".to_string(), s.p10);
    out.insert("p50".to_string(), s.p50);
    out.insert("p90".to_string(), s.p90);
    out.insert("max".to_string(), s.max);
    out.insert("mean".to_string(), s.mean);
    out.insert("mad".to_string(), s.mad);
    if let Some(cov) = s.cov {
        out.insert("cov".to_string(), cov);
    }
    Ok(out)
}

/// Coefficient of variation of per-pass medians, or `None` with fewer than two
/// finite passes / a zero mean.
#[pyfunction]
fn cross_pass_cov(pass_medians: Vec<f64>) -> Option<f64> {
    stats::cross_pass_cov(&pass_medians)
}

/// The requested quantiles (each `q` in `0.0..=1.0`) of a raw sample vector.
/// Raises ``ValueError`` if `samples` is empty / non-finite, or a `q` is out of
/// range.
#[pyfunction]
fn quantiles(samples: Vec<f64>, qs: Vec<f64>) -> PyResult<Vec<f64>> {
    stats::quantiles(&samples, &qs).ok_or_else(|| {
        PyValueError::new_err(
            "samples must be non-empty and finite, and each quantile in 0.0..=1.0",
        )
    })
}

// --- warm-up ------------------------------------------------------------------

/// Find the first warm sample index for `times`. Returns `(start, converged)`.
#[pyfunction]
#[pyo3(signature = (times, window=20, tol=0.02, min_warm=30))]
fn steady_state_index(times: Vec<f64>, window: usize, tol: f64, min_warm: usize) -> (usize, bool) {
    let w = warmup::steady_state(
        &times,
        warmup::WarmupOpts {
            window,
            tol,
            min_warm,
        },
    );
    (w.start, w.converged)
}

// --- bench ----------------------------------------------------------------

/// Run `bench()` against a recorded device session. `opts_json` is a JSON
/// [`BenchOpts`]. Returns the assembled record as canonical JSON. Raises
/// ``ValueError`` on bad options, a malformed recording, or a reduction failure.
#[pyfunction]
fn bench_replay(recording: &str, opts_json: &str) -> PyResult<String> {
    let opts: BenchOpts = serde_json::from_str(opts_json)
        .map_err(|e| PyValueError::new_err(format!("invalid bench options: {e}")))?;
    let record = run_replay(recording, &opts).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(schema::to_json(&record))
}

/// Run `bench()` against a recorded session and return the per-launch
/// GPU-event-time quantiles (each `q` in `0.0..=1.0`), in microseconds -- the
/// raw material for `caliper.do_bench(quantiles=...)`. Raises ``ValueError`` on
/// bad options / recording / quantile.
#[pyfunction]
fn bench_replay_quantiles(
    recording: &str,
    opts_json: &str,
    quantiles: Vec<f64>,
) -> PyResult<Vec<f64>> {
    if let Some(&bad) = quantiles.iter().find(|&&q| !(0.0..=1.0).contains(&q)) {
        return Err(PyValueError::new_err(format!(
            "quantile must be in 0.0..=1.0, got {bad}"
        )));
    }
    let opts: BenchOpts = serde_json::from_str(opts_json)
        .map_err(|e| PyValueError::new_err(format!("invalid bench options: {e}")))?;
    let (_record, qs) = caliper_gpu::run_replay_quantiles(recording, &opts, &quantiles)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(qs)
}

// --- ptxas parsing ------------------------------------------------------

/// Parse a `ptxas -v` / `cuobjdump -res-usage` / HIP `-v` report, sniffing the
/// format. Returns a JSON list of per-kernel resource usage. Raises
/// ``ValueError`` if the text is empty / unrecognised / has no kernels.
#[pyfunction]
fn parse_ptxas(text: &str) -> PyResult<String> {
    let kernels =
        caliper_core::parse_ptxas(text).map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&kernels).map_err(|e| PyValueError::new_err(e.to_string()))
}

// --- occupancy -----------------------------------------------------------

/// Theoretical occupancy for a kernel on `arch` from the CUDA occupancy model.
/// Returns `{theoretical, active_warps_per_sm, active_blocks_per_sm, limiter}`
/// as JSON, or `None` if `arch` is not in the model's table. Raises
/// ``ValueError`` if `threads_per_block` is 0 or above 1024, or
/// `regs_per_thread` is 0 or above 255.
#[pyfunction]
fn theoretical_occupancy(
    arch: &str,
    regs_per_thread: u32,
    smem_bytes_per_block: u32,
    threads_per_block: u32,
) -> PyResult<Option<String>> {
    if threads_per_block == 0 || threads_per_block > 1024 {
        return Err(PyValueError::new_err(
            "threads_per_block must be in 1..=1024",
        ));
    }
    if regs_per_thread == 0 || regs_per_thread > 255 {
        return Err(PyValueError::new_err("regs_per_thread must be in 1..=255"));
    }
    Ok(caliper_core::occupancy::theoretical_occupancy(
        arch,
        regs_per_thread,
        smem_bytes_per_block,
        threads_per_block,
    )
    .map(|est| serde_json::to_string(&est).expect("OccupancyEstimate serialises")))
}

// --- roofline ----------------------------------------------------------------

/// Run the roofline model. Returns the analysis as JSON: `achieved_tflops`,
/// `achieved_gbps`, `arithmetic_intensity`, `ridge_point`, `roofline_pct`,
/// `bound`, and the `peak_*` ceilings used.
#[pyfunction]
fn roofline_analyze(arch: &str, dtype: &str, flops: f64, bytes_hbm: f64, seconds: f64) -> String {
    let spec = caliper_core::roofline::RooflineSpec {
        dtype: dtype.to_string(),
        flops,
        bytes_hbm,
    };
    let r = caliper_core::roofline::analyze(arch, &spec, seconds);
    serde_json::to_string(&r).expect("RooflineResult serialises")
}

/// The dtype-aware compute ceiling (TFLOP/s) for `arch`, or `None` if that
/// architecture/dtype pair is not in the peaks table.
#[pyfunction]
fn peak_compute_tflops(arch: &str, dtype: &str) -> Option<f64> {
    caliper_core::roofline::peak_compute_tflops(arch, dtype)
}

/// The datasheet HBM bandwidth ceiling (GB/s) for `arch`, or `None` if unknown.
#[pyfunction]
fn peak_hbm_gbps(arch: &str) -> Option<f64> {
    caliper_core::roofline::peak_hbm_gbps(arch)
}

/// The inferred `RooflineSpec` (as JSON) for a built-in corpus kernel at
/// `shape_json` (a JSON object of dimensions) and `dtype`, or `None` if the
/// kernel has no roofline or a dimension is missing. Raises ``ValueError`` if
/// `shape_json` is not a JSON object.
#[pyfunction]
#[pyo3(signature = (kernel_key, shape_json, dtype=None))]
fn corpus_roofline_spec(
    kernel_key: &str,
    shape_json: &str,
    dtype: Option<&str>,
) -> PyResult<Option<String>> {
    let shape: caliper_core::schema::JsonMap = serde_json::from_str(shape_json)
        .map_err(|e| PyValueError::new_err(format!("invalid shape: {e}")))?;
    match caliper_core::roofline::corpus_spec(kernel_key, &shape, dtype) {
        None => Ok(None),
        Some(spec) => serde_json::to_string(&spec)
            .map(Some)
            .map_err(|e| PyValueError::new_err(format!("cannot serialise roofline spec: {e}"))),
    }
}

// --- corpus targets ------------------------------------------------------

/// The kernel key for a `corpus:*` target, or `None` if it is not a known
/// oracle. `caliper bench corpus:o1` uses this.
#[pyfunction]
fn resolve_corpus_target(name: &str) -> Option<&'static str> {
    corpus::resolve(name)
}

/// Every built-in `corpus:*` target (oracles and reference kernels) as
/// `(target, kernel_key, description)` triples.
#[pyfunction]
fn corpus_targets() -> Vec<(String, String, String)> {
    corpus::all_targets()
        .into_iter()
        .map(|(t, k, d)| (t.to_string(), k.to_string(), d.to_string()))
        .collect()
}

// --- doctor / fingerprint --------------------------------------------------

/// Run `caliper doctor` against a recorded device session; returns the report
/// as JSON. Raises ``ValueError`` if the recording is malformed.
#[pyfunction]
fn doctor_replay(recording: &str) -> PyResult<String> {
    let mut layer =
        FixturePlayer::from_jsonl(recording).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(gpu_doctor::run(&mut layer).to_json())
}

/// The same, rendered for a terminal (the canonical human format).
#[pyfunction]
fn doctor_render_replay(recording: &str) -> PyResult<String> {
    let mut layer =
        FixturePlayer::from_jsonl(recording).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(gpu_doctor::run(&mut layer).render())
}

/// Run `caliper doctor` against the backend selected by `CALIPER_GPU_PORTS`
/// (default `real`; without the `cuda` build feature that is "no device").
#[pyfunction]
fn doctor_from_env() -> String {
    doctor_report_from_env().to_json()
}

/// The same, rendered for a terminal.
#[pyfunction]
fn doctor_render_from_env() -> String {
    doctor_report_from_env().render()
}

fn doctor_report_from_env() -> caliper_core::doctor::DoctorReport {
    match caliper_gpu::open_from_env() {
        Ok(mut handle) => gpu_doctor::run(&mut handle),
        Err(_) => caliper_core::doctor::DoctorReport::no_device(),
    }
}

/// The machine fingerprint from a recorded session, as JSON. Raises
/// ``ValueError`` if the recording is malformed or has no device snapshot.
#[pyfunction]
fn fingerprint_replay(recording: &str) -> PyResult<String> {
    let mut layer =
        FixturePlayer::from_jsonl(recording).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let machine = layer
        .snapshot()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&machine).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The machine fingerprint from the `CALIPER_GPU_PORTS` backend. Raises
/// ``ValueError`` when there is no device.
#[pyfunction]
fn fingerprint_from_env() -> PyResult<String> {
    let mut handle =
        caliper_gpu::open_from_env().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let machine = handle
        .snapshot()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&machine).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Completeness check for a machine-fingerprint JSON document. Returns
/// `{complete, missing_required, missing_recommended}` as JSON. Raises
/// ``ValueError`` if the input is not a machine document.
#[pyfunction]
fn fingerprint_check(machine_json: &str) -> PyResult<String> {
    let m = caliper_core::fingerprint::from_json(machine_json)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let c = caliper_core::fingerprint::check(&m);
    Ok(serde_json::to_string(&c).expect("FingerprintCheck serialises"))
}

/// `True` iff every required fingerprint field is present. Raises
/// ``ValueError`` if the input is not a machine document.
#[pyfunction]
fn fingerprint_is_complete(machine_json: &str) -> PyResult<bool> {
    let m = caliper_core::fingerprint::from_json(machine_json)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(caliper_core::fingerprint::is_complete(&m))
}

// --- sweep spec ---------------------------------------------------------

/// Validate a `sweep` spec (as JSON -- the Python layer converts the YAML) and
/// return the expanded, deduplicated cell list as a JSON array. Raises
/// ``ValueError`` with a typed message on any malformed field.
#[pyfunction]
fn expand_spec(spec_json: &str) -> PyResult<String> {
    let cells =
        caliper_core::spec::expand(spec_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(serde_json::to_string(&cells).expect("Vec<Cell> serialises"))
}

/// Given the JSON cell list and a JSON array of completed cell keys, return the
/// cells still to run (`--resume`). Raises ``ValueError`` on malformed input.
#[pyfunction]
fn spec_pending(cells_json: &str, done_keys_json: &str) -> PyResult<String> {
    let cells: Vec<caliper_core::spec::Cell> = serde_json::from_str(cells_json)
        .map_err(|e| PyValueError::new_err(format!("invalid cells: {e}")))?;
    let done: Vec<String> = serde_json::from_str(done_keys_json)
        .map_err(|e| PyValueError::new_err(format!("invalid done keys: {e}")))?;
    let left = caliper_core::spec::pending(&cells, &done);
    Ok(serde_json::to_string(&left).expect("Vec<Cell> serialises"))
}

// --- shape libraries ------------------------------------------------------

/// The concrete shape list for a named library (`"square-pow2"`, `"prime-odd"`,
/// `"llm-7b"`, `"llm-70b"`), as a JSON array; `None` for an unknown name.
#[pyfunction]
fn resolve_shape_library(name: &str) -> Option<String> {
    caliper_core::shapes::resolve(name)
        .map(|shapes| serde_json::to_string(&shapes).expect("Vec<Shape> serialises"))
}

/// Every shape-library name.
#[pyfunction]
fn shape_library_names() -> Vec<&'static str> {
    caliper_core::shapes::LIBRARY_NAMES.to_vec()
}

// --- selftest --------------------------------------------------------------

/// Run `caliper selftest` against the `CALIPER_GPU_PORTS` backend and return
/// the Appendix-E report as JSON. With no device this is the `ERROR` /
/// no-device report (exit 2). `full` toggles the O5 + `nsys` cross-checks; the
/// on-device oracle runner itself lands on a CUDA host.
#[pyfunction]
fn selftest_from_env(full: bool) -> String {
    use caliper_core::selftest::{
        SelftestCheck, SelftestReport, CHECK_NAMES, NOT_VALIDATED_TOKENS,
    };

    // A usable device means `snapshot()` succeeds. Without one (no `cuda`
    // build, no GPU) fall to the no-device report -- matches `fingerprint`.
    let machine = match caliper_gpu::open_from_env().and_then(|mut h| h.snapshot()) {
        Ok(m) => m,
        Err(_) => return SelftestReport::no_device(full).to_json(),
    };

    // A device is present, but the on-device oracle runner lands on a CUDA
    // host. Until it does, every oracle is skipped -- so this is an `ERROR`
    // (nothing was validated), honestly.
    let mut checks = vec![SelftestCheck::pass(
        "device_present",
        "a CUDA device is present",
    )];
    for name in CHECK_NAMES {
        if !full && matches!(*name, "o5_cublas_gemm" | "vs_nsys") {
            continue;
        }
        checks.push(SelftestCheck::skip(
            name,
            "on-device oracle runner runs on a CUDA host",
        ));
    }
    SelftestReport::assemble(
        machine,
        checks,
        NOT_VALIDATED_TOKENS
            .iter()
            .map(ToString::to_string)
            .collect(),
    )
    .to_json()
}

/// Assemble a selftest report from a machine document, a JSON array of check
/// objects, and a JSON array of `not_validated` capability tokens. Raises
/// ``ValueError`` on malformed input.
#[pyfunction]
#[pyo3(signature = (machine_json, checks_json, not_validated_json="[]"))]
fn selftest_assemble(
    machine_json: &str,
    checks_json: &str,
    not_validated_json: &str,
) -> PyResult<String> {
    let machine: schema::Machine = serde_json::from_str(machine_json)
        .map_err(|e| PyValueError::new_err(format!("invalid machine: {e}")))?;
    let checks: Vec<caliper_core::selftest::SelftestCheck> = serde_json::from_str(checks_json)
        .map_err(|e| PyValueError::new_err(format!("invalid checks: {e}")))?;
    let not_validated: Vec<String> = serde_json::from_str(not_validated_json)
        .map_err(|e| PyValueError::new_err(format!("invalid not_validated: {e}")))?;
    Ok(caliper_core::selftest::SelftestReport::assemble(machine, checks, not_validated).to_json())
}

/// Structural validation of a selftest report document. Returns a list of
/// problems (empty if well-formed). Raises ``ValueError`` on a parse error.
#[pyfunction]
fn validate_selftest_json(text: &str) -> PyResult<Vec<String>> {
    let report: caliper_core::selftest::SelftestReport =
        serde_json::from_str(text).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(report.validate())
}

/// The CUDA toolchain version in `nvcc --version` output, e.g. `"12.4.131"`, or
/// `None` if the text carries no recognisable version.
#[pyfunction]
fn parse_nvcc_version(output: &str) -> Option<String> {
    caliper_core::fingerprint::parse_nvcc_version(output)
}

/// The CUDA toolchain version in `ptxas --version` output. Same shape as
/// [`parse_nvcc_version`].
#[pyfunction]
fn parse_ptxas_version(output: &str) -> Option<String> {
    caliper_core::fingerprint::parse_ptxas_version(output)
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", schema::CALIPER_VERSION)?;
    module.add_function(wrap_pyfunction!(schema_version, module)?)?;
    module.add_function(wrap_pyfunction!(core_version, module)?)?;
    module.add_function(wrap_pyfunction!(default_record_json, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_record_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_record_json, module)?)?;
    module.add_function(wrap_pyfunction!(summarize, module)?)?;
    module.add_function(wrap_pyfunction!(cross_pass_cov, module)?)?;
    module.add_function(wrap_pyfunction!(quantiles, module)?)?;
    module.add_function(wrap_pyfunction!(steady_state_index, module)?)?;
    module.add_function(wrap_pyfunction!(bench_replay, module)?)?;
    module.add_function(wrap_pyfunction!(bench_replay_quantiles, module)?)?;
    module.add_function(wrap_pyfunction!(parse_ptxas, module)?)?;
    module.add_function(wrap_pyfunction!(theoretical_occupancy, module)?)?;
    module.add_function(wrap_pyfunction!(roofline_analyze, module)?)?;
    module.add_function(wrap_pyfunction!(peak_compute_tflops, module)?)?;
    module.add_function(wrap_pyfunction!(peak_hbm_gbps, module)?)?;
    module.add_function(wrap_pyfunction!(corpus_roofline_spec, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_corpus_target, module)?)?;
    module.add_function(wrap_pyfunction!(corpus_targets, module)?)?;
    module.add_function(wrap_pyfunction!(doctor_replay, module)?)?;
    module.add_function(wrap_pyfunction!(doctor_render_replay, module)?)?;
    module.add_function(wrap_pyfunction!(doctor_from_env, module)?)?;
    module.add_function(wrap_pyfunction!(doctor_render_from_env, module)?)?;
    module.add_function(wrap_pyfunction!(fingerprint_replay, module)?)?;
    module.add_function(wrap_pyfunction!(fingerprint_from_env, module)?)?;
    module.add_function(wrap_pyfunction!(fingerprint_check, module)?)?;
    module.add_function(wrap_pyfunction!(fingerprint_is_complete, module)?)?;
    module.add_function(wrap_pyfunction!(expand_spec, module)?)?;
    module.add_function(wrap_pyfunction!(spec_pending, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_shape_library, module)?)?;
    module.add_function(wrap_pyfunction!(shape_library_names, module)?)?;
    module.add_function(wrap_pyfunction!(selftest_from_env, module)?)?;
    module.add_function(wrap_pyfunction!(selftest_assemble, module)?)?;
    module.add_function(wrap_pyfunction!(validate_selftest_json, module)?)?;
    module.add_function(wrap_pyfunction!(parse_nvcc_version, module)?)?;
    module.add_function(wrap_pyfunction!(parse_ptxas_version, module)?)?;
    Ok(())
}
