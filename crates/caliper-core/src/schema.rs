//! The result schema: caliper's single output record.
//!
//! Every command that measures something emits a [`Record`] (or a table of
//! them). This module is the reference (de)serialisation and validation for that
//! record; the Python layer and the on-disk format both defer to it.
//!
//! Design notes:
//!
//! * Every field is optional. Modules fill in the sections they own as the
//!   pipeline is built; a partially populated record is still valid.
//! * Deserialisation is lenient: unknown keys are ignored and missing sections
//!   fall back to their defaults, so a record written by an older build still
//!   loads.
//! * [`to_json`] is canonical: fields serialise in declaration order and map
//!   keys are sorted, so `normalize_json(to_json(r)) == to_json(r)`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The schema version embedded in every [`Record`]. Bumped only on a
/// breaking change to the field set or semantics.
pub const SCHEMA_VERSION: &str = "1";

/// The `caliper-core` crate version, recorded in every [`Record`].
pub const CALIPER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// A free-form JSON object (used for kernel shapes and autotune configs).
pub type JsonMap = BTreeMap<String, Value>;

fn default_schema_version() -> String {
    SCHEMA_VERSION.to_string()
}

fn default_caliper_version() -> String {
    CALIPER_VERSION.to_string()
}

/// Identity of the thing being measured.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct KernelLabel {
    /// Kernel or symbol name.
    pub name: Option<String>,
    /// Implementation family: `"triton"`, `"cuda"`, `"cublas"`, `"torch"`, ...
    pub r#impl: Option<String>,
    /// Content hash of the kernel source, for pinning.
    pub source_hash: Option<String>,
    /// The autotune configuration this measurement used.
    pub autotune_config: JsonMap,
    /// Element dtype, e.g. `"bf16"`, `"fp8_e4m3"`.
    pub dtype: Option<String>,
    /// Problem shape, e.g. `{"M": 4096, "N": 4096, "K": 4096}`.
    pub shape: JsonMap,
    /// Memory layout, e.g. `"row"`, `"col"`, `"strided"`.
    pub layout: Option<String>,
}

/// Wall-clock and GPU-event timing, reported as a distribution.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Timing {
    /// 10th-percentile GPU-event time (microseconds).
    pub p10_us: Option<f64>,
    /// Median GPU-event time (microseconds).
    pub p50_us: Option<f64>,
    /// 90th-percentile GPU-event time (microseconds).
    pub p90_us: Option<f64>,
    /// Arithmetic mean of the per-launch GPU-event times (microseconds). This
    /// is what a Triton-style `do_bench(return_mode="mean")` reports.
    pub mean_us: Option<f64>,
    /// Fastest per-launch GPU-event time (microseconds).
    pub min_us: Option<f64>,
    /// Slowest per-launch GPU-event time (microseconds).
    pub max_us: Option<f64>,
    /// Median absolute deviation of the samples (microseconds).
    pub mad_us: Option<f64>,
    /// Median wall-clock time including host-side overhead (microseconds).
    pub wall_p50_us: Option<f64>,
    /// Estimated CPU-side launch overhead per invocation (microseconds).
    pub launch_overhead_us: Option<f64>,
    /// Number of kept samples.
    pub n_samples: Option<u64>,
    /// Iterations discarded before the clock reached steady state.
    pub n_warmup_to_steady: Option<u64>,
    /// Samples thrown away because the GPU was throttling.
    pub invalidated_samples: Option<u64>,
    /// Coefficient of variation of p50 across measurement passes.
    pub cross_pass_cov: Option<f64>,
}

/// Achieved throughput relative to the hardware roofline.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Roofline {
    /// Achieved compute throughput (TFLOP/s).
    pub achieved_tflops: Option<f64>,
    /// Fraction of the dtype's roofline peak achieved (1.0 == at the roofline).
    pub roofline_pct: Option<f64>,
    /// Achieved memory throughput (GB/s).
    pub achieved_gbps: Option<f64>,
    /// FLOPs per byte of HBM traffic.
    pub arithmetic_intensity: Option<f64>,
    /// Arithmetic intensity at which the roofline changes slope.
    pub ridge_point: Option<f64>,
    /// `"compute"`, `"memory"`, `"latency"`, or `"unknown"`.
    pub bound: Option<String>,
    /// Achieved throughput as a fraction of the vendor library baseline.
    pub baseline_pct: Option<f64>,
    /// Which baseline was compared against: `"cublas"`, `"cudnn"`, `"torch"`, ...
    pub baseline: Option<String>,
}

/// Static resource usage reported by the compiler (`ptxas -v` / `cuobjdump`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Ptxas {
    /// Registers per thread.
    pub regs_per_thread: Option<u32>,
    /// Statically allocated shared memory (bytes).
    pub smem_static_bytes: Option<u64>,
    /// Dynamically allocated shared memory (bytes).
    pub smem_dynamic_bytes: Option<u64>,
    /// Bytes loaded from local memory due to register spills.
    pub spill_loads_bytes: Option<u64>,
    /// Bytes stored to local memory due to register spills.
    pub spill_stores_bytes: Option<u64>,
    /// Local memory per thread (bytes).
    pub local_bytes: Option<u64>,
    /// Stack frame per thread (bytes).
    pub stack_bytes: Option<u64>,
}

/// Theoretical and achieved occupancy.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Occupancy {
    /// Theoretical occupancy from the CUDA occupancy model (0..=1).
    pub theoretical: Option<f64>,
    /// Achieved occupancy observed during the run (0..=1).
    pub achieved: Option<f64>,
    /// Active warps per SM.
    pub active_warps_per_sm: Option<u32>,
    /// Number of scheduling waves for the launch.
    pub waves: Option<f64>,
}

/// GPU clock state during the measurement.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Clocks {
    /// SM clock (MHz).
    pub sm_mhz: Option<u32>,
    /// Memory clock (MHz).
    pub mem_mhz: Option<u32>,
    /// Whether clocks were locked for the measurement.
    pub locked: Option<bool>,
    /// How clocks were locked, e.g. `"nvml"`; `None` if not locked.
    pub lock_method: Option<String>,
}

/// Versions of the compilation / kernel toolchain.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Toolkit {
    /// Triton version.
    pub triton: Option<String>,
    /// PyTorch version.
    pub torch: Option<String>,
    /// `ptxas` version.
    pub ptxas: Option<String>,
    /// `nvcc` version.
    pub nvcc: Option<String>,
}

/// Everything about the host and device needed to interpret a record.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Machine {
    /// Marketing name, e.g. `"NVIDIA GeForce RTX 4090"`.
    pub gpu_name: Option<String>,
    /// Compute capability tag, e.g. `"sm_89"`.
    pub sm_arch: Option<String>,
    /// Total device memory (MiB).
    pub vram_mib: Option<u64>,
    /// Number of streaming multiprocessors.
    pub sm_count: Option<u32>,
    /// L2 cache size (bytes) -- used to size the cache-flush buffer.
    pub l2_bytes: Option<u64>,
    /// BAR1 aperture size (MiB).
    pub bar1_mib: Option<u64>,
    /// Driver version.
    pub driver: Option<String>,
    /// CUDA runtime version.
    pub cuda_runtime: Option<String>,
    /// CUDA driver API version.
    pub cuda_driver: Option<String>,
    /// NVML version.
    pub nvml_version: Option<String>,
    /// Whether ECC is enabled.
    pub ecc: Option<bool>,
    /// MIG state: `"disabled"` or a geometry string.
    pub mig: Option<String>,
    /// Whether persistence mode is on.
    pub persistence_mode: Option<bool>,
    /// PCIe generation.
    pub pcie_gen: Option<u32>,
    /// PCIe link width (lanes).
    pub pcie_width: Option<u32>,
    /// Toolchain versions.
    pub toolkit: Toolkit,
}

/// One measurement, with all the context needed to interpret it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Record {
    /// Schema version; see [`SCHEMA_VERSION`].
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    /// Version of `caliper-core` that produced this record.
    #[serde(default = "default_caliper_version")]
    pub caliper_version: String,
    /// ISO-8601 UTC timestamp of the measurement.
    pub measured_at: Option<String>,
    /// Salted, non-identifying host id (for de-duplicating community submissions).
    pub host_id_class: Option<String>,
    /// What was measured.
    pub kernel: KernelLabel,
    /// Timing distribution.
    pub timing: Timing,
    /// Roofline analysis.
    pub roofline: Roofline,
    /// Compiler resource usage.
    pub ptxas: Ptxas,
    /// Occupancy.
    pub occupancy: Occupancy,
    /// Clock state.
    pub clocks: Clocks,
    /// Host and device description.
    pub machine: Machine,
    /// Throttle reasons observed during the run (NVML names).
    pub throttle_reasons: Vec<String>,
    /// Free-form advisory tags, e.g. `"clocks-unlocked"`.
    pub flags: Vec<String>,
}

impl Default for Record {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            caliper_version: default_caliper_version(),
            measured_at: None,
            host_id_class: None,
            kernel: KernelLabel::default(),
            timing: Timing::default(),
            roofline: Roofline::default(),
            ptxas: Ptxas::default(),
            occupancy: Occupancy::default(),
            clocks: Clocks::default(),
            machine: Machine::default(),
            throttle_reasons: Vec::new(),
            flags: Vec::new(),
        }
    }
}

/// Parse a [`Record`] from JSON. Lenient: unknown keys are ignored and missing
/// sections use their defaults.
///
/// # Errors
/// Returns the `serde_json` error if `s` is not valid JSON, or has a value of
/// the wrong type for a known field.
pub fn from_json(s: &str) -> Result<Record, serde_json::Error> {
    serde_json::from_str(s)
}

/// Serialise a [`Record`] to canonical, compact JSON (stable field and key
/// order).
///
/// # Panics
/// Never in practice: a [`Record`] holds only JSON-representable values, so
/// `serde_json` serialisation cannot fail.
pub fn to_json(record: &Record) -> String {
    serde_json::to_string(record).expect("Record always serialises")
}

/// Serialise a [`Record`] to canonical, indented JSON.
///
/// # Panics
/// Never in practice; see [`to_json`].
pub fn to_json_pretty(record: &Record) -> String {
    serde_json::to_string_pretty(record).expect("Record always serialises")
}

/// Parse then re-serialise, producing the canonical form of an arbitrary
/// (possibly hand-written or older-schema) record document.
///
/// # Errors
/// Returns the `serde_json` error if `s` does not parse as a record.
pub fn normalize_json(s: &str) -> Result<String, serde_json::Error> {
    Ok(to_json(&from_json(s)?))
}

/// Check a record for internal inconsistencies. Returns a list of human-readable
/// problems; an empty list means the record is well-formed.
///
/// This is the seed of the `caliper validate` command. It grows as sections gain
/// meaning; today it covers the invariants that hold regardless of hardware.
pub fn validate(record: &Record) -> Vec<String> {
    let mut problems = Vec::new();

    if record.schema_version != SCHEMA_VERSION {
        problems.push(format!(
            "unsupported schema_version {:?}; this build understands {SCHEMA_VERSION:?}",
            record.schema_version
        ));
    }

    check_finite_nonneg("timing.p10_us", record.timing.p10_us, &mut problems);
    check_finite_nonneg("timing.p50_us", record.timing.p50_us, &mut problems);
    check_finite_nonneg("timing.p90_us", record.timing.p90_us, &mut problems);
    check_finite_nonneg("timing.mean_us", record.timing.mean_us, &mut problems);
    check_finite_nonneg("timing.min_us", record.timing.min_us, &mut problems);
    check_finite_nonneg("timing.max_us", record.timing.max_us, &mut problems);
    check_finite_nonneg("timing.mad_us", record.timing.mad_us, &mut problems);
    check_finite_nonneg(
        "timing.wall_p50_us",
        record.timing.wall_p50_us,
        &mut problems,
    );

    if let (Some(p10), Some(p50)) = (record.timing.p10_us, record.timing.p50_us) {
        if p10 > p50 {
            problems.push(format!(
                "timing.p10_us ({p10}) is greater than timing.p50_us ({p50})"
            ));
        }
    }
    if let (Some(p50), Some(p90)) = (record.timing.p50_us, record.timing.p90_us) {
        if p50 > p90 {
            problems.push(format!(
                "timing.p50_us ({p50}) is greater than timing.p90_us ({p90})"
            ));
        }
    }

    if let Some(pct) = record.roofline.roofline_pct {
        if !(0.0..=1.5).contains(&pct) || !pct.is_finite() {
            problems.push(format!(
                "roofline.roofline_pct ({pct}) is outside the plausible range 0.0..=1.5"
            ));
        }
    }
    if let Some(occ) = record.occupancy.theoretical {
        if !(0.0..=1.0).contains(&occ) || !occ.is_finite() {
            problems.push(format!(
                "occupancy.theoretical ({occ}) is outside 0.0..=1.0"
            ));
        }
    }
    if let Some(occ) = record.occupancy.achieved {
        if !(0.0..=1.0).contains(&occ) || !occ.is_finite() {
            problems.push(format!("occupancy.achieved ({occ}) is outside 0.0..=1.0"));
        }
    }

    problems
}

/// Parse a record document and validate it. Parse errors surface as `Err`;
/// schema problems come back in the `Ok` list.
///
/// # Errors
/// Returns the `serde_json` error if `s` does not parse as a record.
pub fn validate_json(s: &str) -> Result<Vec<String>, serde_json::Error> {
    Ok(validate(&from_json(s)?))
}

fn check_finite_nonneg(field: &str, value: Option<f64>, problems: &mut Vec<String>) {
    if let Some(x) = value {
        if !x.is_finite() || x < 0.0 {
            problems.push(format!(
                "{field} ({x}) is not a finite, non-negative number"
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_carries_versions_and_empty_sections() {
        let r = Record::default();
        assert_eq!(r.schema_version, SCHEMA_VERSION);
        assert!(!r.caliper_version.is_empty());
        assert!(r.throttle_reasons.is_empty());
        assert!(r.timing.p50_us.is_none());
    }

    #[test]
    fn default_round_trips() {
        let r = Record::default();
        let json = to_json(&r);
        assert_eq!(from_json(&json).unwrap(), r);
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)] // readable test setup
    fn populated_record_round_trips_and_is_canonical() {
        let mut r = Record::default();
        r.measured_at = Some("2026-01-02T03:04:05Z".to_string());
        r.host_id_class = Some("sha256:abc".to_string());
        r.kernel.name = Some("matmul_kernel".to_string());
        r.kernel.r#impl = Some("triton".to_string());
        r.kernel
            .autotune_config
            .insert("BLOCK_M".to_string(), serde_json::json!(128));
        r.kernel
            .autotune_config
            .insert("num_warps".to_string(), serde_json::json!(8));
        r.kernel
            .shape
            .insert("M".to_string(), serde_json::json!(4096));
        r.kernel.dtype = Some("bf16".to_string());
        r.timing.p10_us = Some(241.0);
        r.timing.p50_us = Some(243.2);
        r.timing.p90_us = Some(250.1);
        r.timing.n_samples = Some(300);
        r.roofline.achieved_tflops = Some(565.0);
        r.roofline.roofline_pct = Some(0.86);
        r.roofline.bound = Some("compute".to_string());
        r.ptxas.regs_per_thread = Some(168);
        r.occupancy.theoretical = Some(0.25);
        r.clocks.sm_mhz = Some(2520);
        r.clocks.locked = Some(true);
        r.machine.sm_arch = Some("sm_89".to_string());
        r.machine.toolkit.triton = Some("3.2.0".to_string());
        r.throttle_reasons.push("SW_POWER_CAP".to_string());
        r.flags.push("clocks-unlocked".to_string());

        let json = to_json(&r);
        assert_eq!(from_json(&json).unwrap(), r);
        assert_eq!(normalize_json(&json).unwrap(), json);
    }

    #[test]
    fn deserialisation_ignores_unknown_keys_and_fills_missing_sections() {
        let json = r#"{"kernel":{"impl":"cuda","not_a_field":1},"surprise":true}"#;
        let r: Record = from_json(json).unwrap();
        assert_eq!(r.kernel.r#impl.as_deref(), Some("cuda"));
        assert_eq!(r.schema_version, SCHEMA_VERSION);
        assert!(r.timing.p50_us.is_none());
        assert!(!to_json(&r).contains("not_a_field"));
    }

    #[test]
    fn missing_version_falls_back_to_current() {
        let r: Record = from_json(r#"{"kernel":{"impl":"cuda"}}"#).unwrap();
        assert_eq!(r.schema_version, SCHEMA_VERSION);
        assert_eq!(r.caliper_version, CALIPER_VERSION);
    }

    #[test]
    fn validate_accepts_a_default_record() {
        assert!(validate(&Record::default()).is_empty());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)] // readable test setup
    fn validate_flags_each_class_of_problem() {
        let mut r = Record::default();
        r.schema_version = "99".to_string();
        r.timing.p10_us = Some(9.0);
        r.timing.p50_us = Some(2.0);
        r.roofline.roofline_pct = Some(4.0);
        r.occupancy.achieved = Some(-0.1);

        let problems = validate(&r);
        assert_eq!(problems.len(), 4, "problems were: {problems:#?}");
    }

    #[test]
    fn validate_json_surfaces_parse_errors() {
        assert!(validate_json("{not valid json").is_err());
    }
}
