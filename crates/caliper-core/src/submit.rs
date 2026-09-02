//! Building and validating a `caliper submit` results bundle.
//!
//! A bundle is `manifest.json` + a rows file (`rows.parquet` / `.jsonl`) +
//! `fingerprint.json`. [`derive_manifest`] summarises a row set into a
//! [`Manifest`]; [`validate_bundle`] is the shared gate both this repo and
//! `caliper-results` run -- schema validity, the submission-strict field and
//! roofline checks, and the bundle-level determinism / calibration / arch
//! consistency checks.
//!
//! The `toolchain_hash` is computed by the caller (Appendix C: sha256 of the
//! sorted toolkit map + driver) -- `caliper-core` carries no hash dependency.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::schema::{Machine, Record, SCHEMA_VERSION};
use crate::stats::cross_pass_cov;

/// Fields a row must carry to be submitted to `caliper-results` (the schema
/// itself keeps everything optional so a half-built record still round-trips).
pub const SUBMISSION_REQUIRED: &[&str] = &[
    "measured_at",
    "kernel.name",
    "kernel.dtype",
    "timing.p50_us",
    "timing.n_samples",
    "machine.sm_arch",
];

/// A submitted row may not claim more than this fraction of the dtype roofline
/// peak. The schema's own 1.5 clamp allows measurement noise; a *submission*
/// past ~100% is a mislabelled peak or a bad FLOP count.
pub const SUBMISSION_MAX_ROOFLINE_PCT: f64 = 1.05;

/// CoV(p50) a determinism repeat must stay under, per NFR-5.
pub const DETERMINISM_TOL_LOCKED: f64 = 0.02;
/// CoV(p50) a determinism repeat must stay under on the unlocked (Colab) tier.
pub const DETERMINISM_TOL_UNLOCKED: f64 = 0.05;
/// How far the calibration GEMM's p50 may sit from its per-SKU expectation.
pub const CALIBRATION_TOL: f64 = 0.08;

/// The clock-lock tier a bundle was measured on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// NVML clock lock held for every measurement.
    Locked,
    /// Clocks were free (the Colab default); wider CoV is tolerated.
    Unlocked,
}

/// The O7 calibration-GEMM check carried in a bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub measured_p50_us: f64,
    pub expected_p50_us: f64,
    /// `measured / expected`.
    pub ratio: f64,
    pub within_tolerance: bool,
}

impl Calibration {
    #[must_use]
    pub fn new(measured_p50_us: f64, expected_p50_us: f64) -> Self {
        let ratio = if expected_p50_us > 0.0 {
            measured_p50_us / expected_p50_us
        } else {
            f64::INFINITY
        };
        Self {
            measured_p50_us,
            expected_p50_us,
            ratio,
            within_tolerance: (ratio - 1.0).abs() <= CALIBRATION_TOL,
        }
    }
}

/// A determinism repeat: several measurements of one facet, and how tightly
/// their medians agreed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Determinism {
    pub facet: String,
    pub n_repeats: usize,
    /// CoV of the repeat p50s.
    pub cov: f64,
    pub tolerance: f64,
    pub within_tolerance: bool,
}

/// The summary that rides at the top of a bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub caliper_version: String,
    pub schema_version: String,
    pub created_at: String,
    /// `machine.sm_arch`, consistent across every row.
    pub arch: String,
    /// Appendix-C partition key (computed by the caller).
    pub toolchain_hash: String,
    pub n_rows: usize,
    /// Sorted distinct `kernel.name`.
    pub kernels: Vec<String>,
    pub tier: Tier,
    pub calibration: Option<Calibration>,
    pub determinism: Option<Determinism>,
}

/// Everything that can go wrong assembling a manifest from a row set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BundleError {
    /// The row set was empty.
    NoRows,
    /// Rows carry more than one `machine.sm_arch` (or none).
    ArchMismatch(Vec<String>),
}

impl std::fmt::Display for BundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoRows => write!(f, "no rows to submit"),
            Self::ArchMismatch(a) => {
                write!(f, "rows must share one machine.sm_arch, found {a:?}")
            }
        }
    }
}

impl std::error::Error for BundleError {}

fn facet_of(r: &Record) -> String {
    format!(
        "{}|{}|{}|{}",
        r.kernel.name.as_deref().unwrap_or(""),
        r.kernel.dtype.as_deref().unwrap_or(""),
        serde_json::to_string(&r.kernel.shape).unwrap_or_default(),
        r.kernel.layout.as_deref().unwrap_or(""),
    )
}

/// The tier a row set was measured on: `Unlocked` if any row is flagged
/// `clocks-unlocked`, else `Locked`.
#[must_use]
pub fn tier_of(rows: &[Record]) -> Tier {
    if rows
        .iter()
        .any(|r| r.flags.iter().any(|f| f == "clocks-unlocked"))
    {
        Tier::Unlocked
    } else {
        Tier::Locked
    }
}

/// The largest determinism repeat in `rows`: the facet with the most rows that
/// carry a `p50`, when that count is at least two.
#[must_use]
pub fn largest_repeat(rows: &[Record], tier: Tier) -> Option<Determinism> {
    let mut by_facet: std::collections::BTreeMap<String, Vec<f64>> =
        std::collections::BTreeMap::new();
    for r in rows {
        if let Some(p50) = r.timing.p50_us {
            by_facet.entry(facet_of(r)).or_default().push(p50);
        }
    }
    let (facet, p50s) = by_facet.into_iter().max_by_key(|(_, v)| v.len())?;
    if p50s.len() < 2 {
        return None;
    }
    let tolerance = match tier {
        Tier::Locked => DETERMINISM_TOL_LOCKED,
        Tier::Unlocked => DETERMINISM_TOL_UNLOCKED,
    };
    let cov = cross_pass_cov(&p50s).unwrap_or(f64::INFINITY);
    Some(Determinism {
        facet,
        n_repeats: p50s.len(),
        cov,
        tolerance,
        within_tolerance: cov <= tolerance,
    })
}

/// Summarise a row set into a [`Manifest`]. `calibration` is `(measured,
/// expected)` p50 microseconds for the SKU's calibration GEMM, when the
/// submitter ran one.
///
/// # Errors
/// [`BundleError::NoRows`] for an empty set, [`BundleError::ArchMismatch`] when
/// the rows do not share exactly one `machine.sm_arch`.
pub fn derive_manifest(
    rows: &[Record],
    toolchain_hash: &str,
    caliper_version: &str,
    created_at: &str,
    calibration: Option<(f64, f64)>,
) -> Result<Manifest, BundleError> {
    if rows.is_empty() {
        return Err(BundleError::NoRows);
    }
    let arches: BTreeSet<String> = rows
        .iter()
        .map(|r| r.machine.sm_arch.clone().unwrap_or_default())
        .collect();
    if arches.len() != 1 || arches.contains("") {
        return Err(BundleError::ArchMismatch(arches.into_iter().collect()));
    }
    let arch = arches.into_iter().next().unwrap();
    let kernels: Vec<String> = rows
        .iter()
        .filter_map(|r| r.kernel.name.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let tier = tier_of(rows);

    Ok(Manifest {
        caliper_version: caliper_version.to_string(),
        schema_version: SCHEMA_VERSION.to_string(),
        created_at: created_at.to_string(),
        arch,
        toolchain_hash: toolchain_hash.to_string(),
        n_rows: rows.len(),
        kernels,
        tier,
        calibration: calibration.map(|(m, e)| Calibration::new(m, e)),
        determinism: largest_repeat(rows, tier),
    })
}

fn get_path<'a>(v: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    let mut cur = v;
    for seg in path.split('.') {
        cur = cur.get(seg)?;
    }
    (!cur.is_null()).then_some(cur)
}

/// The submission-strict problems with one row: missing required fields and an
/// over-peak roofline claim. (`schema::validate` covers the hardware-independent
/// invariants; this is the extra bar for a public submission.)
#[must_use]
pub fn submission_row_problems(row: &serde_json::Value) -> Vec<String> {
    let mut problems = Vec::new();
    for field in SUBMISSION_REQUIRED {
        if get_path(row, field).is_none() {
            problems.push(format!("missing required field {field}"));
        }
    }
    if let Some(pct) = get_path(row, "roofline.roofline_pct").and_then(serde_json::Value::as_f64) {
        if pct > SUBMISSION_MAX_ROOFLINE_PCT {
            problems.push(format!(
                "roofline.roofline_pct ({pct}) claims more than {SUBMISSION_MAX_ROOFLINE_PCT} of peak"
            ));
        }
    }
    problems
}

/// Validate a bundle: the shared gate for `caliper validate <dir>` and the
/// `caliper-results` CI. Returns a list of human-readable problems (empty when
/// the bundle is clean).
///
/// # Errors
/// A `serde_json` error if any input is not the expected JSON shape.
pub fn validate_bundle(
    manifest_json: &str,
    rows_json: &str,
    fingerprint_json: &str,
) -> Result<Vec<String>, serde_json::Error> {
    let manifest: Manifest = serde_json::from_str(manifest_json)?;
    let rows: Vec<serde_json::Value> = serde_json::from_str(rows_json)?;
    let fingerprint: Machine = serde_json::from_str(fingerprint_json)?;

    let mut problems = Vec::new();

    if manifest.schema_version != SCHEMA_VERSION {
        problems.push(format!(
            "manifest schema_version {:?}; this build understands {SCHEMA_VERSION:?}",
            manifest.schema_version
        ));
    }
    if manifest.n_rows != rows.len() {
        problems.push(format!(
            "manifest n_rows ({}) != rows file length ({})",
            manifest.n_rows,
            rows.len()
        ));
    }
    if fingerprint.sm_arch.as_deref() != Some(manifest.arch.as_str()) {
        problems.push(format!(
            "fingerprint.sm_arch ({:?}) != manifest arch ({:?})",
            fingerprint.sm_arch, manifest.arch
        ));
    }

    for (i, row) in rows.iter().enumerate() {
        let text = row.to_string();
        match crate::schema::validate_json(&text) {
            Ok(schema_problems) => {
                for p in schema_problems {
                    problems.push(format!("row {i}: {p}"));
                }
            }
            Err(e) => problems.push(format!("row {i}: does not parse as a record: {e}")),
        }
        for p in submission_row_problems(row) {
            problems.push(format!("row {i}: {p}"));
        }
        let row_arch = get_path(row, "machine.sm_arch").and_then(serde_json::Value::as_str);
        if row_arch.is_some() && row_arch != Some(manifest.arch.as_str()) {
            problems.push(format!(
                "row {i}: machine.sm_arch ({row_arch:?}) != manifest arch ({:?})",
                manifest.arch
            ));
        }
    }

    if let Some(c) = &manifest.calibration {
        if !c.within_tolerance {
            problems.push(format!(
                "calibration GEMM p50 is {:.1}% of expected ({:.1} vs {:.1} us; tolerance +/-{:.0}%)",
                c.ratio * 100.0,
                c.measured_p50_us,
                c.expected_p50_us,
                CALIBRATION_TOL * 100.0
            ));
        }
    }
    if let Some(d) = &manifest.determinism {
        if !d.within_tolerance {
            problems.push(format!(
                "determinism repeat CoV {:.1}% exceeds the {:.1}% tolerance for facet {}",
                d.cov * 100.0,
                d.tolerance * 100.0,
                d.facet
            ));
        }
    }

    Ok(problems)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::field_reassign_with_default)] // readable test setup
    fn row(name: &str, dtype: &str, p50: f64, arch: &str) -> Record {
        let mut r = Record::default();
        r.measured_at = Some("2026-09-04T00:00:00Z".into());
        r.kernel.name = Some(name.into());
        r.kernel.dtype = Some(dtype.into());
        r.kernel.layout = Some("row".into());
        r.timing.p50_us = Some(p50);
        r.timing.n_samples = Some(300);
        r.machine.sm_arch = Some(arch.into());
        r
    }

    fn rows_json(rows: &[Record]) -> String {
        serde_json::to_string(
            &rows
                .iter()
                .map(|r| serde_json::to_value(r).unwrap())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    fn fingerprint(arch: &str) -> String {
        serde_json::json!({ "sm_arch": arch }).to_string()
    }

    #[test]
    fn derive_manifest_summarises_arch_kernels_and_tier() {
        let mut rows = vec![
            row("corpus:gemm", "bf16", 243.0, "sm_80"),
            row("corpus:rmsnorm", "bf16", 60.0, "sm_80"),
        ];
        rows[0].flags.push("clocks-unlocked".into());
        let m = derive_manifest(&rows, "abc123", "0.3.0", "2026-09-04T00:00:00Z", None).unwrap();
        assert_eq!(m.arch, "sm_80");
        assert_eq!(m.kernels, ["corpus:gemm", "corpus:rmsnorm"]);
        assert_eq!(m.tier, Tier::Unlocked);
        assert_eq!(m.n_rows, 2);
        assert!(m.determinism.is_none()); // no facet repeated
    }

    #[test]
    fn derive_manifest_rejects_an_empty_or_mixed_arch_set() {
        assert_eq!(
            derive_manifest(&[], "h", "v", "t", None),
            Err(BundleError::NoRows)
        );
        let rows = [
            row("corpus:gemm", "bf16", 1.0, "sm_80"),
            row("corpus:gemm", "bf16", 1.0, "sm_90"),
        ];
        assert!(matches!(
            derive_manifest(&rows, "h", "v", "t", None),
            Err(BundleError::ArchMismatch(_))
        ));
    }

    #[test]
    fn a_repeated_facet_becomes_a_determinism_block() {
        let rows: Vec<Record> = [243.0, 244.5, 242.8, 243.9]
            .iter()
            .map(|&p| {
                let mut r = row("corpus:gemm", "bf16", p, "sm_80");
                r.flags.push("clocks-unlocked".into());
                r
            })
            .collect();
        let d = derive_manifest(&rows, "h", "v", "t", None)
            .unwrap()
            .determinism
            .unwrap();
        assert_eq!(d.n_repeats, 4);
        assert!(d.cov < 0.05 && d.within_tolerance);
        assert_eq!(d.tolerance, DETERMINISM_TOL_UNLOCKED);
    }

    #[test]
    fn calibration_tolerance_is_plus_minus_eight_percent() {
        assert!(Calibration::new(104.0, 100.0).within_tolerance); // +4%
        assert!(!Calibration::new(115.0, 100.0).within_tolerance); // +15%
        assert!((Calibration::new(115.0, 100.0).ratio - 1.15).abs() < 1e-9);
    }

    #[test]
    fn a_clean_bundle_has_no_problems() {
        let rows = [
            row("corpus:gemm", "bf16", 243.0, "sm_80"),
            row("corpus:rmsnorm", "bf16", 60.0, "sm_80"),
        ];
        let m = derive_manifest(&rows, "abc", "0.3.0", "t", Some((101.0, 100.0))).unwrap();
        let problems = validate_bundle(
            &serde_json::to_string(&m).unwrap(),
            &rows_json(&rows),
            &fingerprint("sm_80"),
        )
        .unwrap();
        assert_eq!(problems, Vec::<String>::new());
    }

    #[test]
    fn a_missing_required_field_is_rejected() {
        let mut bad = row("corpus:gemm", "bf16", 243.0, "sm_80");
        bad.kernel.name = None;
        let rows = [bad];
        let m = derive_manifest(
            &[row("corpus:gemm", "bf16", 243.0, "sm_80")],
            "abc",
            "0.3.0",
            "t",
            None,
        )
        .unwrap();
        let problems = validate_bundle(
            &serde_json::to_string(&m).unwrap(),
            &rows_json(&rows),
            &fingerprint("sm_80"),
        )
        .unwrap();
        assert!(problems
            .iter()
            .any(|p| p.contains("missing required field kernel.name")));
    }

    #[test]
    fn an_over_peak_roofline_claim_is_rejected() {
        let mut bad = row("corpus:gemm", "bf16", 243.0, "sm_80");
        // 1.2 passes the schema's 1.5 clamp but not the submission bar.
        bad.roofline.roofline_pct = Some(1.2);
        let rows = [bad];
        let m = derive_manifest(&rows, "abc", "0.3.0", "t", None).unwrap();
        let problems = validate_bundle(
            &serde_json::to_string(&m).unwrap(),
            &rows_json(&rows),
            &fingerprint("sm_80"),
        )
        .unwrap();
        assert!(problems.iter().any(|p| p.contains("claims more than")));
    }

    #[test]
    fn a_nonreproducing_determinism_repeat_is_rejected() {
        let rows: Vec<Record> = [243.0, 300.0, 210.0, 275.0]
            .iter()
            .map(|&p| {
                let mut r = row("corpus:gemm", "bf16", p, "sm_80");
                r.flags.push("clocks-unlocked".into());
                r
            })
            .collect();
        let m = derive_manifest(&rows, "abc", "0.3.0", "t", None).unwrap();
        assert!(!m.determinism.as_ref().unwrap().within_tolerance);
        let problems = validate_bundle(
            &serde_json::to_string(&m).unwrap(),
            &rows_json(&rows),
            &fingerprint("sm_80"),
        )
        .unwrap();
        assert!(problems
            .iter()
            .any(|p| p.contains("determinism repeat CoV")));
    }

    #[test]
    fn a_slow_calibration_kernel_is_rejected() {
        let rows = [row("corpus:gemm", "bf16", 243.0, "sm_80")];
        let m = derive_manifest(&rows, "abc", "0.3.0", "t", Some((118.0, 100.0))).unwrap();
        let problems = validate_bundle(
            &serde_json::to_string(&m).unwrap(),
            &rows_json(&rows),
            &fingerprint("sm_80"),
        )
        .unwrap();
        assert!(problems
            .iter()
            .any(|p| p.contains("calibration GEMM p50 is")));
    }

    #[test]
    fn a_fingerprint_or_row_on_the_wrong_arch_is_rejected() {
        let rows = [row("corpus:gemm", "bf16", 243.0, "sm_80")];
        let m = derive_manifest(&rows, "abc", "0.3.0", "t", None).unwrap();
        let problems = validate_bundle(
            &serde_json::to_string(&m).unwrap(),
            &rows_json(&rows),
            &fingerprint("sm_90"),
        )
        .unwrap();
        assert!(problems.iter().any(|p| p.contains("fingerprint.sm_arch")));
    }
}
