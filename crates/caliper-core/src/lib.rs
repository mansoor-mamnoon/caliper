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
pub mod fingerprint;
pub mod graph;
pub mod occupancy;
pub mod oracles;
pub mod pipeline;
pub mod ptxas_parse;
pub mod roofline;
pub mod schema;
pub mod stats;
pub mod warmup;

pub use doctor::{assess, DoctorFacts, DoctorReport};
pub use fingerprint::{
    assert_complete as assert_fingerprint_complete, check as check_fingerprint, FingerprintCheck,
    FingerprintError,
};
pub use graph::{resolve as resolve_graph_mode, GraphChoice};
pub use occupancy::{theoretical_occupancy, Limiter, OccupancyEstimate};
pub use oracles::{fit_line, LineFit, OracleCheck};
pub use pipeline::{
    flush_buffer_bytes, invalidate, reduce, reduce_quantiles, PipelineError, ReduceInput,
};
pub use ptxas_parse::{parse_any as parse_ptxas, ParsedKernel, PtxasParseError};
pub use roofline::{
    analyze as roofline_analyze, corpus_spec as corpus_roofline_spec, peak_compute_tflops,
    peak_fp32_fma_tflops, peak_hbm_gbps, peak_tensor_tflops, Bound, RooflineResult, RooflineSpec,
};
pub use schema::{Record, SCHEMA_VERSION};
pub use stats::{cross_pass_cov, summarize, Summary};
pub use warmup::{steady_state, Warmup, WarmupOpts, WarmupPlan};
