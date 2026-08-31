//! PyO3 bindings for `caliper-core`.
//!
//! This crate is a thin marshalling layer: it converts between Python values and
//! `caliper-core` calls, and does no logic of its own. It is built by maturin
//! into the `caliper._core` extension module.

use std::collections::HashMap;

use caliper_core::{schema, stats, warmup};
use caliper_gpu::{run_replay, BenchOpts};
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
    Ok(())
}
