//! Parsing and expanding a `sweep` spec (Appendix D).
//!
//! The spec is authored as YAML; the Python layer reads the file and hands this
//! module the JSON form. [`expand`] validates it and produces the flat list of
//! [`Cell`]s a sweep runs -- the cartesian product of dtypes, layouts, and
//! resolved shapes -- deduplicated by [`Cell::key`]. [`pending`] drops the cells
//! a `--resume` state file already records as done.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::shapes::{self, Shape};

/// Dtypes a sweep may request.
const KNOWN_DTYPES: &[&str] = &["bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "tf32", "fp32"];
/// Memory layouts a sweep may request.
const KNOWN_LAYOUTS: &[&str] = &["row", "col"];
/// The only schema version this build understands (0 = unset -> treated as 1).
const SUPPORTED_SCHEMA_VERSION: u32 = 1;

fn default_layouts() -> Vec<String> {
    vec!["row".to_string()]
}
fn default_autotune() -> String {
    "from_kernel".to_string()
}

/// `shapes:` is either a library name or an inline list.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ShapesField {
    /// A named library (see [`crate::shapes`]).
    Named(String),
    /// An explicit shape list.
    Inline(Vec<Shape>),
}

/// `bench.warmup` is `"auto"` or a fixed non-negative count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Warmup {
    /// A word, which must be `"auto"`.
    Word(String),
    /// A fixed number of leading samples to trim.
    Fixed(u64),
}

impl Default for Warmup {
    fn default() -> Self {
        Warmup::Word("auto".to_string())
    }
}

/// The `bench:` block: how each cell is measured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BenchParams {
    /// Warm-up policy.
    pub warmup: Warmup,
    /// Minimum kept samples.
    pub min_samples: u64,
    /// Flush the L2 cache between samples.
    pub flush_l2: bool,
    /// Lock the clocks for the run.
    pub lock_clocks: bool,
    /// CUDA-graph policy (`"auto"` / `"on"` / `"off"`).
    pub cuda_graph: String,
}

impl Default for BenchParams {
    fn default() -> Self {
        Self {
            warmup: Warmup::default(),
            min_samples: 200,
            flush_l2: true,
            lock_clocks: true,
            cuda_graph: "auto".to_string(),
        }
    }
}

/// The `output:` block.
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct OutputSpec {
    /// Parquet destination.
    pub parquet: Option<String>,
    /// JSON destination.
    pub json: Option<String>,
    /// Whether to resume from an existing state file.
    pub resume: bool,
}

/// A `sweep` spec, as the YAML deserialises.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepSpec {
    /// Spec schema version (0 or absent -> 1).
    #[serde(default)]
    pub schema_version: u32,
    /// What to sweep: `corpus:*` or `path::kernel`.
    pub target: String,
    /// Element dtypes to cross.
    pub dtypes: Vec<String>,
    /// Memory layouts to cross (defaults to `[row]`).
    #[serde(default = "default_layouts")]
    pub layouts: Vec<String>,
    /// Named library or inline shape list.
    pub shapes: ShapesField,
    /// Per-cell bench parameters.
    #[serde(default)]
    pub bench: BenchParams,
    /// Autotune policy (recorded, not expanded here).
    #[serde(default = "default_autotune")]
    pub autotune: String,
    /// Output destinations.
    #[serde(default)]
    pub output: OutputSpec,
}

/// One measurement the sweep will run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cell {
    /// The sweep target.
    pub target: String,
    /// Element dtype.
    pub dtype: String,
    /// Memory layout.
    pub layout: String,
    /// Concrete problem shape.
    pub shape: Shape,
    /// Bench parameters (identical across a spec's cells).
    pub bench: BenchParams,
}

impl Cell {
    /// A stable identity for dedupe and `--resume`. The bench parameters do not
    /// vary within one spec, so they are not part of the key.
    #[must_use]
    pub fn key(&self) -> String {
        format!(
            "{}|{}|{}|{}",
            self.target,
            self.dtype,
            self.layout,
            self.shape.label()
        )
    }
}

/// Everything that can go wrong parsing or expanding a spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecError {
    /// The JSON did not deserialise into a [`SweepSpec`] (missing field, unknown
    /// key, wrong type).
    Parse(String),
    /// `schema_version` is not supported.
    UnsupportedSchemaVersion(u32),
    /// `target` is empty.
    EmptyTarget,
    /// `dtypes` is empty.
    EmptyDtypes,
    /// `layouts` is empty.
    EmptyLayouts,
    /// A dtype is not one caliper knows.
    UnknownDtype(String),
    /// A layout is not `row` or `col`.
    UnknownLayout(String),
    /// `bench.warmup` was a word other than `"auto"`.
    BadWarmup(String),
    /// `shapes:` named a library that does not exist.
    UnknownShapeLibrary(String),
    /// An inline shape has a non-positive dimension.
    BadShape(String),
    /// The spec expands to zero cells.
    EmptyExpansion,
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "spec does not parse: {e}"),
            Self::UnsupportedSchemaVersion(v) => {
                write!(f, "unsupported schema_version {v}; this build understands {SUPPORTED_SCHEMA_VERSION}")
            }
            Self::EmptyTarget => f.write_str("target is empty"),
            Self::EmptyDtypes => f.write_str("dtypes is empty"),
            Self::EmptyLayouts => f.write_str("layouts is empty"),
            Self::UnknownDtype(d) => write!(f, "unknown dtype {d:?}; known: {KNOWN_DTYPES:?}"),
            Self::UnknownLayout(l) => write!(f, "unknown layout {l:?}; known: {KNOWN_LAYOUTS:?}"),
            Self::BadWarmup(w) => write!(f, "bench.warmup {w:?} must be \"auto\" or a number"),
            Self::UnknownShapeLibrary(n) => write!(
                f,
                "unknown shape library {n:?}; known: {:?}",
                shapes::LIBRARY_NAMES
            ),
            Self::BadShape(s) => write!(f, "shape {s} has a non-positive dimension"),
            Self::EmptyExpansion => f.write_str("the spec expands to zero cells"),
        }
    }
}

impl std::error::Error for SpecError {}

fn shape_is_positive(shape: &Shape) -> bool {
    match *shape {
        Shape::Gemm { m, n, k } => m > 0 && n > 0 && k > 0,
        Shape::Attn { b, h, s, d } => b > 0 && h > 0 && s > 0 && d > 0,
    }
}

/// Validate a JSON spec and expand it to the deduplicated list of cells to run.
///
/// # Errors
/// A [`SpecError`] for any malformed or empty field, or an unknown
/// dtype / layout / shape library.
pub fn expand(spec_json: &str) -> Result<Vec<Cell>, SpecError> {
    let spec: SweepSpec =
        serde_json::from_str(spec_json).map_err(|e| SpecError::Parse(e.to_string()))?;

    if !matches!(spec.schema_version, 0 | SUPPORTED_SCHEMA_VERSION) {
        return Err(SpecError::UnsupportedSchemaVersion(spec.schema_version));
    }
    if spec.target.trim().is_empty() {
        return Err(SpecError::EmptyTarget);
    }
    if spec.dtypes.is_empty() {
        return Err(SpecError::EmptyDtypes);
    }
    if spec.layouts.is_empty() {
        return Err(SpecError::EmptyLayouts);
    }
    for d in &spec.dtypes {
        if !KNOWN_DTYPES.contains(&d.as_str()) {
            return Err(SpecError::UnknownDtype(d.clone()));
        }
    }
    for l in &spec.layouts {
        if !KNOWN_LAYOUTS.contains(&l.as_str()) {
            return Err(SpecError::UnknownLayout(l.clone()));
        }
    }
    if let Warmup::Word(w) = &spec.bench.warmup {
        if w != "auto" {
            return Err(SpecError::BadWarmup(w.clone()));
        }
    }

    let shapes: Vec<Shape> = match &spec.shapes {
        ShapesField::Named(name) => {
            shapes::resolve(name).ok_or_else(|| SpecError::UnknownShapeLibrary(name.clone()))?
        }
        ShapesField::Inline(list) => {
            for s in list {
                if !shape_is_positive(s) {
                    return Err(SpecError::BadShape(s.label()));
                }
            }
            list.clone()
        }
    };

    let mut cells = Vec::new();
    let mut seen = HashSet::new();
    for dtype in &spec.dtypes {
        for layout in &spec.layouts {
            for shape in &shapes {
                let cell = Cell {
                    target: spec.target.clone(),
                    dtype: dtype.clone(),
                    layout: layout.clone(),
                    shape: *shape,
                    bench: spec.bench.clone(),
                };
                if seen.insert(cell.key()) {
                    cells.push(cell);
                }
            }
        }
    }

    if cells.is_empty() {
        return Err(SpecError::EmptyExpansion);
    }
    Ok(cells)
}

/// The cells in `all` whose [`Cell::key`] is not in `done_keys` -- the work a
/// `--resume` still has to do.
#[must_use]
pub fn pending(all: &[Cell], done_keys: &[String]) -> Vec<Cell> {
    let done: HashSet<&str> = done_keys.iter().map(String::as_str).collect();
    all.iter()
        .filter(|c| !done.contains(c.key().as_str()))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `square-pow2` spec with the given `dtypes` (a JSON array literal) and
    /// any extra `"key":value` pairs appended.
    fn spec_json(dtypes: &str, extra: &str) -> String {
        format!(r#"{{"target":"corpus:gemm","dtypes":{dtypes},"shapes":"square-pow2"{extra}}}"#)
    }

    #[test]
    fn a_named_library_expands_to_the_cartesian_product() {
        let cells = expand(&spec_json(
            r#"["bf16","fp16"]"#,
            r#","layouts":["row","col"]"#,
        ))
        .unwrap();
        // 2 dtypes x 2 layouts x 5 square-pow2 shapes
        assert_eq!(cells.len(), 20);
        // deterministic order: dtype outer, then layout, then shape
        assert_eq!(cells[0].dtype, "bf16");
        assert_eq!(cells[0].layout, "row");
        assert_eq!(cells[5].layout, "col");
        // every key is unique
        let keys: HashSet<_> = cells.iter().map(Cell::key).collect();
        assert_eq!(keys.len(), 20);
    }

    #[test]
    fn duplicate_inline_shapes_are_deduped() {
        let json = r#"{"target":"corpus:gemm","dtypes":["bf16"],"shapes":[
            {"kind":"gemm","m":128,"n":128,"k":128},
            {"kind":"gemm","m":128,"n":128,"k":128},
            {"kind":"gemm","m":256,"n":256,"k":256}]}"#;
        let cells = expand(json).unwrap();
        assert_eq!(cells.len(), 2);
    }

    #[test]
    fn defaults_are_filled_in() {
        let cells = expand(&spec_json(r#"["bf16"]"#, "")).unwrap();
        assert_eq!(cells[0].layout, "row"); // default
        assert_eq!(cells[0].bench, BenchParams::default());
        assert_eq!(cells[0].bench.min_samples, 200);
    }

    #[test]
    fn each_bad_field_is_a_typed_error() {
        // unknown key
        assert!(matches!(
            expand(r#"{"target":"x","dtypes":["bf16"],"shapes":"square-pow2","wat":1}"#),
            Err(SpecError::Parse(_))
        ));
        // missing required field
        assert!(matches!(
            expand(r#"{"dtypes":["bf16"],"shapes":"square-pow2"}"#),
            Err(SpecError::Parse(_))
        ));
        assert_eq!(
            expand(&spec_json(r#"["bf17"]"#, "")),
            Err(SpecError::UnknownDtype("bf17".to_string()))
        );
        assert_eq!(
            expand(&spec_json(r#"["bf16"]"#, r#","layouts":["diagonal"]"#)),
            Err(SpecError::UnknownLayout("diagonal".to_string()))
        );
        assert_eq!(expand(&spec_json("[]", "")), Err(SpecError::EmptyDtypes));
        assert_eq!(
            expand(r#"{"target":" ","dtypes":["bf16"],"shapes":"square-pow2"}"#),
            Err(SpecError::EmptyTarget)
        );
        assert_eq!(
            expand(r#"{"target":"x","dtypes":["bf16"],"shapes":"llm-7000b"}"#),
            Err(SpecError::UnknownShapeLibrary("llm-7000b".to_string()))
        );
        assert_eq!(
            expand(&spec_json(r#"["bf16"]"#, r#","schema_version":9"#)),
            Err(SpecError::UnsupportedSchemaVersion(9))
        );
        assert_eq!(
            expand(&spec_json(
                r#"["bf16"]"#,
                r#","bench":{"warmup":"sometimes"}"#
            )),
            Err(SpecError::BadWarmup("sometimes".to_string()))
        );
        assert!(matches!(
            expand(
                r#"{"target":"x","dtypes":["bf16"],"shapes":[{"kind":"gemm","m":0,"n":1,"k":1}]}"#
            ),
            Err(SpecError::BadShape(_))
        ));
    }

    #[test]
    fn warmup_accepts_auto_or_a_number() {
        assert!(expand(&spec_json(r#"["bf16"]"#, r#","bench":{"warmup":"auto"}"#)).is_ok());
        assert!(expand(&spec_json(r#"["bf16"]"#, r#","bench":{"warmup":25}"#)).is_ok());
    }

    #[test]
    fn the_appendix_d_example_expands_to_36_unique_cells() {
        let json = r#"{"schema_version":1,"target":"corpus:gemm",
            "dtypes":["bf16","fp16","fp8_e4m3"],"layouts":["row","col"],
            "shapes":"llm-7b",
            "bench":{"warmup":"auto","min_samples":200,"flush_l2":true,"lock_clocks":true,"cuda_graph":"auto"},
            "autotune":"from_kernel",
            "output":{"parquet":"results/gemm-sweep.parquet","resume":true}}"#;
        let cells = expand(json).unwrap();
        assert_eq!(cells.len(), 3 * 2 * 6);
        let keys: HashSet<_> = cells.iter().map(Cell::key).collect();
        assert_eq!(keys.len(), cells.len());
    }

    #[test]
    fn pending_drops_the_finished_cells() {
        let cells = expand(&spec_json(r#"["bf16"]"#, "")).unwrap();
        let done = vec![cells[0].key(), cells[2].key()];
        let left = pending(&cells, &done);
        assert_eq!(left.len(), cells.len() - 2);
        assert!(!left.iter().any(|c| c.key() == cells[0].key()));
        // resuming a finished sweep leaves nothing
        let all_done: Vec<String> = cells.iter().map(Cell::key).collect();
        assert!(pending(&cells, &all_done).is_empty());
    }
}
