//! PyO3 bindings for `caliper-core`.
//!
//! This crate is a thin marshalling layer: it converts between Python strings /
//! lists and `caliper-core` calls, and does no logic of its own. It is built by
//! maturin into the `caliper._core` extension module.

use caliper_core::schema;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", schema::CALIPER_VERSION)?;
    module.add_function(wrap_pyfunction!(schema_version, module)?)?;
    module.add_function(wrap_pyfunction!(core_version, module)?)?;
    module.add_function(wrap_pyfunction!(default_record_json, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_record_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_record_json, module)?)?;
    Ok(())
}
