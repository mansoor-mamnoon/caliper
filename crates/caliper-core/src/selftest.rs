//! The `caliper selftest` report (Appendix E of the plan).
//!
//! `selftest` runs the on-device oracle checks (O1-O4, O6, O7) plus a
//! reproducibility pass, and `--full` adds O5 and an `nsys` cross-check. Each
//! check contributes a [`SelftestCheck`]; [`SelftestReport::assemble`] folds
//! them into an overall `PASS` / `FAIL` / `ERROR`, a `full` / `reduced`
//! coverage, and the list of checks that could not be validated here.
//!
//! The oracle *execution* runs on a CUDA host; this module is the pure report
//! model and is fully `cargo test`-covered. On a machine with no device the
//! CLI emits [`SelftestReport::no_device`] (`ERROR`, exit 2).

use serde::{Deserialize, Serialize};

use crate::oracles::OracleCheck;
use crate::schema::{JsonMap, Machine, CALIPER_VERSION, SCHEMA_VERSION};

/// Outcome of a single check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CheckStatus {
    /// Ran and met its expectation.
    Pass,
    /// Ran and missed its expectation.
    Fail,
    /// Could not run here (no device, tool missing, on-device path deferred).
    Skip,
    /// Errored while running.
    Error,
}

/// Overall selftest result. Maps to exit code 0 / 1 / 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Outcome {
    /// At least one check passed and none failed or errored.
    Pass,
    /// A check failed (and none errored).
    Fail,
    /// A check errored, or nothing was actually validated.
    Error,
}

/// How much of the suite ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Coverage {
    /// The `nsys` cross-check ran (implies `--full` on a host with `nsys`).
    Full,
    /// No `nsys` cross-check.
    Reduced,
}

/// One line of the selftest report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelftestCheck {
    /// Stable check name, e.g. `"o1_duration_linearity"`.
    pub name: String,
    /// Whether it passed / failed / skipped / errored.
    pub status: CheckStatus,
    /// What was measured (free-form; e.g. `{"slope": 1.006}`).
    #[serde(default, skip_serializing_if = "JsonMap::is_empty")]
    pub measured: JsonMap,
    /// What first principles say it should be.
    #[serde(default, skip_serializing_if = "JsonMap::is_empty")]
    pub expected: JsonMap,
    /// Tolerance applied, as a human string (e.g. `"3%"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<String>,
    /// Human-readable explanation.
    pub detail: String,
}

impl SelftestCheck {
    /// A check that ran and passed, with no numeric measurement to record.
    #[must_use]
    pub fn pass(name: &str, detail: &str) -> Self {
        Self {
            name: name.to_string(),
            status: CheckStatus::Pass,
            measured: JsonMap::new(),
            expected: JsonMap::new(),
            tolerance: None,
            detail: detail.to_string(),
        }
    }

    /// A check that could not run here.
    #[must_use]
    pub fn skip(name: &str, detail: &str) -> Self {
        Self {
            name: name.to_string(),
            status: CheckStatus::Skip,
            measured: JsonMap::new(),
            expected: JsonMap::new(),
            tolerance: None,
            detail: detail.to_string(),
        }
    }

    /// A check that errored while running.
    #[must_use]
    pub fn error(name: &str, detail: &str) -> Self {
        Self {
            name: name.to_string(),
            status: CheckStatus::Error,
            measured: JsonMap::new(),
            expected: JsonMap::new(),
            tolerance: None,
            detail: detail.to_string(),
        }
    }

    /// Build a report line from an [`OracleCheck`] the on-device runner produced.
    #[must_use]
    pub fn from_oracle(name: &str, check: &OracleCheck) -> Self {
        let mut measured = JsonMap::new();
        measured.insert("value".to_string(), serde_json::json!(check.measured));
        let mut expected = JsonMap::new();
        expected.insert("value".to_string(), serde_json::json!(check.expected));
        let tolerance = (check.tolerance > 0.0).then(|| format!("{:.1}%", check.tolerance * 100.0));
        Self {
            name: name.to_string(),
            status: if check.passed {
                CheckStatus::Pass
            } else {
                CheckStatus::Fail
            },
            measured,
            expected,
            tolerance,
            detail: check.detail.clone(),
        }
    }
}

/// The names of every check the suite reports, in order. The on-device runner
/// fills these in; anything it cannot run stays `Skip` and lands in
/// `not_validated`.
pub const CHECK_NAMES: &[&str] = &[
    "o1_duration_linearity",
    "o2_bandwidth",
    "o2_flush_ab",
    "o3_fma_peak",
    "o4_launch_overhead",
    "o6_throttle",
    "o7_calibration_gemm",
    "reproducibility",
    "o5_cublas_gemm", // --full only
    "vs_nsys",        // --full only
];

/// The assembled `caliper selftest` report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelftestReport {
    /// Result-schema version.
    pub schema_version: String,
    /// `caliper` version that produced this report.
    pub caliper_version: String,
    /// The machine the suite ran on.
    pub machine: Machine,
    /// Overall `PASS` / `FAIL` / `ERROR`.
    pub result: Outcome,
    /// `full` (nsys cross-check ran) or `reduced`.
    pub coverage: Coverage,
    /// Every check, in the order it ran.
    pub checks: Vec<SelftestCheck>,
    /// Names of checks that did not validate here (all `Skip` checks).
    pub not_validated: Vec<String>,
}

impl SelftestReport {
    /// Fold a set of checks into a report: derive the overall result, the
    /// coverage, and the `not_validated` list.
    ///
    /// The result is `ERROR` if any check errored **or if no check actually
    /// passed** (a selftest that validated nothing is not a pass); otherwise
    /// `FAIL` if any check failed; otherwise `PASS`.
    #[must_use]
    pub fn assemble(machine: Machine, checks: Vec<SelftestCheck>) -> Self {
        let any_error = checks.iter().any(|c| c.status == CheckStatus::Error);
        let any_fail = checks.iter().any(|c| c.status == CheckStatus::Fail);
        let any_pass = checks.iter().any(|c| c.status == CheckStatus::Pass);

        let result = if any_error || !any_pass {
            Outcome::Error
        } else if any_fail {
            Outcome::Fail
        } else {
            Outcome::Pass
        };

        let coverage = if checks
            .iter()
            .any(|c| c.name == "vs_nsys" && c.status == CheckStatus::Pass)
        {
            Coverage::Full
        } else {
            Coverage::Reduced
        };

        let not_validated = checks
            .iter()
            .filter(|c| c.status == CheckStatus::Skip)
            .map(|c| c.name.clone())
            .collect();

        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            caliper_version: CALIPER_VERSION.to_string(),
            machine,
            result,
            coverage,
            checks,
            not_validated,
        }
    }

    /// The report for a host with no CUDA device: a `device_present` error plus
    /// every oracle check skipped. `ERROR`, exit 2.
    #[must_use]
    pub fn no_device() -> Self {
        let mut checks = vec![SelftestCheck::error(
            "device_present",
            "no CUDA device found; the oracle suite cannot run",
        )];
        checks.extend(
            CHECK_NAMES
                .iter()
                .map(|n| SelftestCheck::skip(n, "no CUDA device")),
        );
        Self::assemble(Machine::default(), checks)
    }

    /// Canonical JSON.
    ///
    /// # Panics
    /// Never in practice: the report holds only JSON-representable values.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("SelftestReport always serialises")
    }

    /// Process exit code: `PASS` -> 0, `FAIL` -> 1, `ERROR` -> 2.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self.result {
            Outcome::Pass => 0,
            Outcome::Fail => 1,
            Outcome::Error => 2,
        }
    }

    /// Structural self-consistency check: the derived fields must match the
    /// check list. An empty list means the report is well-formed.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut problems = Vec::new();

        if self.schema_version != SCHEMA_VERSION {
            problems.push(format!(
                "schema_version {:?} != {SCHEMA_VERSION:?}",
                self.schema_version
            ));
        }
        if self.checks.is_empty() {
            problems.push("report has no checks".to_string());
        }
        if self.checks.iter().any(|c| c.name.trim().is_empty()) {
            problems.push("a check has an empty name".to_string());
        }

        let recomputed = Self::assemble(self.machine.clone(), self.checks.clone());
        if recomputed.result != self.result {
            problems.push(format!(
                "result {:?} does not follow from the checks (expected {:?})",
                self.result, recomputed.result
            ));
        }
        if recomputed.coverage != self.coverage {
            problems.push(format!(
                "coverage {:?} does not follow from the checks (expected {:?})",
                self.coverage, recomputed.coverage
            ));
        }
        if recomputed.not_validated != self.not_validated {
            problems.push("not_validated does not match the SKIP checks".to_string());
        }

        problems
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(name: &str) -> SelftestCheck {
        SelftestCheck::pass(name, "ok")
    }
    fn fail(name: &str) -> SelftestCheck {
        SelftestCheck {
            status: CheckStatus::Fail,
            ..SelftestCheck::pass(name, "missed")
        }
    }

    #[test]
    fn all_pass_is_a_pass_exit_zero() {
        let r = SelftestReport::assemble(
            Machine::default(),
            vec![pass("o1_duration_linearity"), pass("o3_fma_peak")],
        );
        assert_eq!(r.result, Outcome::Pass);
        assert_eq!(r.coverage, Coverage::Reduced);
        assert_eq!(r.exit_code(), 0);
        assert!(r.not_validated.is_empty());
        assert!(r.validate().is_empty());
    }

    #[test]
    fn a_failing_check_makes_the_result_fail() {
        let r = SelftestReport::assemble(
            Machine::default(),
            vec![pass("o1_duration_linearity"), fail("o3_fma_peak")],
        );
        assert_eq!(r.result, Outcome::Fail);
        assert_eq!(r.exit_code(), 1);
    }

    #[test]
    fn an_erroring_check_beats_a_failing_one() {
        let r = SelftestReport::assemble(
            Machine::default(),
            vec![fail("a"), SelftestCheck::error("b", "boom")],
        );
        assert_eq!(r.result, Outcome::Error);
        assert_eq!(r.exit_code(), 2);
    }

    #[test]
    fn a_suite_that_validated_nothing_is_an_error_not_a_pass() {
        let r = SelftestReport::assemble(
            Machine::default(),
            vec![
                SelftestCheck::skip("o1_duration_linearity", "no device"),
                SelftestCheck::skip("o3_fma_peak", "no device"),
            ],
        );
        assert_eq!(r.result, Outcome::Error);
        assert_eq!(r.exit_code(), 2);
        assert_eq!(r.not_validated.len(), 2);
    }

    #[test]
    fn nsys_pass_lifts_coverage_to_full() {
        let r = SelftestReport::assemble(
            Machine::default(),
            vec![pass("o1_duration_linearity"), pass("vs_nsys")],
        );
        assert_eq!(r.coverage, Coverage::Full);

        let skipped = SelftestReport::assemble(
            Machine::default(),
            vec![
                pass("o1_duration_linearity"),
                SelftestCheck::skip("vs_nsys", "nsys not on PATH"),
            ],
        );
        assert_eq!(skipped.coverage, Coverage::Reduced);
        assert_eq!(skipped.not_validated, vec!["vs_nsys"]);
    }

    #[test]
    fn no_device_report_is_error_exit_two_and_lists_everything() {
        let r = SelftestReport::no_device();
        assert_eq!(r.result, Outcome::Error);
        assert_eq!(r.exit_code(), 2);
        assert_eq!(r.coverage, Coverage::Reduced);
        assert_eq!(r.not_validated.len(), CHECK_NAMES.len());
        assert!(r.checks.iter().any(|c| c.name == "device_present"));
        assert!(r.validate().is_empty());
    }

    #[test]
    fn report_round_trips_and_validate_catches_tampering() {
        let r = SelftestReport::assemble(
            Machine::default(),
            vec![pass("o1_duration_linearity"), pass("o2_bandwidth")],
        );
        let json = r.to_json();
        let back: SelftestReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);

        let mut tampered = r.clone();
        tampered.result = Outcome::Fail;
        assert!(!tampered.validate().is_empty());

        let mut bad_cov = r;
        bad_cov.coverage = Coverage::Full;
        assert!(!bad_cov.validate().is_empty());
    }

    #[test]
    fn from_oracle_maps_pass_fail_and_tolerance() {
        let check = OracleCheck {
            name: "o3_fma".to_string(),
            passed: true,
            measured: 300.0,
            expected: 312.0,
            tolerance: 0.10,
            detail: "96% of peak".to_string(),
        };
        let line = SelftestCheck::from_oracle("o3_fma_peak", &check);
        assert_eq!(line.status, CheckStatus::Pass);
        assert_eq!(line.tolerance.as_deref(), Some("10.0%"));
        assert_eq!(line.measured["value"], serde_json::json!(300.0));
    }
}
