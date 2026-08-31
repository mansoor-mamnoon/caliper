//! Pure measurement logic for caliper.
//!
//! This crate has no GPU dependency and no Python dependency. Everything here is
//! deterministic and covered by `cargo test`; together with the Python tests
//! that exercise it through the bindings, it is caliper's no-GPU ("L0") surface.
//!
//! The measurement engine (statistics, steady-state detection, the reduction
//! pipeline, the roofline model, `ptxas` parsing, the regression threshold
//! model, sweep-spec expansion) lands here module by module.

pub mod doctor;
pub mod oracles;
pub mod pipeline;
pub mod schema;
pub mod stats;
pub mod warmup;

pub use doctor::{assess, DoctorFacts, DoctorReport};
pub use oracles::{fit_line, LineFit, OracleCheck};
pub use pipeline::{flush_buffer_bytes, invalidate, reduce, PipelineError, ReduceInput};
pub use schema::{Record, SCHEMA_VERSION};
pub use stats::{cross_pass_cov, summarize, Summary};
pub use warmup::{steady_state, Warmup, WarmupOpts, WarmupPlan};
