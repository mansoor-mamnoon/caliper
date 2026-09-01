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

// --- corpus targets ------------------------------------------------------

/// The kernel key for a `corpus:*` target, or `None` if it is not a known
/// oracle. `caliper bench corpus:o1` uses this.
#[pyfunction]
fn resolve_corpus_target(name: &str) -> Option<&'static str> {
    corpus::resolve(name)
}

/// The built-in oracle targets as `(target, kernel_key, description)` triples.
#[pyfunction]
fn corpus_targets() -> Vec<(String, String, String)> {
    corpus::ORACLE_TARGETS
        .iter()
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
    module.add_function(wrap_pyfunction!(steady_state_index, module)?)?;
    module.add_function(wrap_pyfunction!(bench_replay, module)?)?;
    module.add_function(wrap_pyfunction!(parse_ptxas, module)?)?;
    module.add_function(wrap_pyfunction!(theoretical_occupancy, module)?)?;
    module.add_function(wrap_pyfunction!(roofline_analyze, module)?)?;
    module.add_function(wrap_pyfunction!(peak_compute_tflops, module)?)?;
    module.add_function(wrap_pyfunction!(peak_hbm_gbps, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_corpus_target, module)?)?;
    module.add_function(wrap_pyfunction!(corpus_targets, module)?)?;
    module.add_function(wrap_pyfunction!(doctor_replay, module)?)?;
    module.add_function(wrap_pyfunction!(doctor_render_replay, module)?)?;
    module.add_function(wrap_pyfunction!(doctor_from_env, module)?)?;
    module.add_function(wrap_pyfunction!(doctor_render_from_env, module)?)?;
    module.add_function(wrap_pyfunction!(fingerprint_replay, module)?)?;
    module.add_function(wrap_pyfunction!(fingerprint_from_env, module)?)?;
    Ok(())
}
