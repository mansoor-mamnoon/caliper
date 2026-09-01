//! The `caliper selftest` report (Appendix E of the plan).
//!
//! `selftest` runs the on-device oracle checks (O1-O4, O6, O7) plus a
//! reproducibility pass, and `--full` adds O5 (cuBLAS) and an `nsys`
//! cross-check. Each contributes a [`SelftestCheck`];
//! [`SelftestReport::assemble`] folds them into an overall `PASS` / `FAIL` /
//! `ERROR`, a `full` / `reduced` coverage, and the `not_validated` list of
//! capabilities a constrained host could not exercise.
//!
//! ## Result rules (plan §3.4)
//!
//! * Only the suite checks in [`CHECK_NAMES`] are *scored*; context lines like
//!   `device_present` are reported but do not count.
//! * `ERROR` if any scored check errored, or if **no** scored check actually
//!   passed (a run that validated nothing is not a pass).
//! * otherwise `FAIL` if any scored check failed;
//! * otherwise `PASS` -- including a `reduced`-coverage run, as long as every
//!   non-`SKIP` check passed.
//!
//! ## The on-device runner
//!
//! Executing the oracles needs a CUDA host: run `bench("corpus:oN", ...)`, feed
//! the `Result` to the matching `oracles::check_*`, and turn each into a
//! [`SelftestCheck`] via [`SelftestCheck::from_oracle`] (or [`SelftestCheck::skip`]
//! for one that cannot run here). Then [`SelftestReport::assemble`]. This module
//! is the pure report side and is fully `cargo test`-covered; with no device
//! the CLI emits [`SelftestReport::no_device`] (`ERROR`, exit 2).

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
    /// Every scored check passed (at least one did) and none failed or errored.
    Pass,
    /// A scored check failed (and none errored).
    Fail,
    /// A scored check errored, or nothing was actually validated.
    Error,
}

/// How much of the suite ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Coverage {
    /// The `nsys` cross-check ran.
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
    /// What was measured (free-form; e.g. `{"value": 1.006}`).
    #[serde(default, skip_serializing_if = "JsonMap::is_empty")]
    pub measured: JsonMap,
    /// What first principles say it should be.
    #[serde(default, skip_serializing_if = "JsonMap::is_empty")]
    pub expected: JsonMap,
    /// Tolerance applied, as a human string (e.g. `"3.0%"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<String>,
    /// Human-readable explanation.
    pub detail: String,
}

impl SelftestCheck {
    /// A check that ran and passed, with no numeric measurement to record.
    #[must_use]
    pub fn pass(name: &str, detail: &str) -> Self {
        Self::bare(name, CheckStatus::Pass, detail)
    }

    /// A check that could not run here.
    #[must_use]
    pub fn skip(name: &str, detail: &str) -> Self {
        Self::bare(name, CheckStatus::Skip, detail)
    }

    /// A check that errored while running.
    #[must_use]
    pub fn error(name: &str, detail: &str) -> Self {
        Self::bare(name, CheckStatus::Error, detail)
    }

    fn bare(name: &str, status: CheckStatus, detail: &str) -> Self {
        Self {
            name: name.to_string(),
            status,
            measured: JsonMap::new(),
            expected: JsonMap::new(),
            tolerance: None,
            detail: detail.to_string(),
        }
    }

    /// Build a report line from an [`OracleCheck`] the on-device runner
    /// produced. `name` is the [`CHECK_NAMES`] entry (the oracle check's own
    /// name is often shorter, e.g. `o3_fma` -> `o3_fma_peak`).
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

/// The scored suite checks, in report order. `--full` adds the last two.
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

/// The capability tokens a constrained host reports in `not_validated`
/// (plan §0.5 / §3.4). No other value is allowed there.
pub const NOT_VALIDATED_TOKENS: &[&str] = &["clock_lock", "ncu_crosscheck", "powercap_throttle"];

fn is_scored(name: &str) -> bool {
    CHECK_NAMES.contains(&name)
}

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
    /// Capability tokens that could not be validated here -- a subset of
    /// [`NOT_VALIDATED_TOKENS`].
    pub not_validated: Vec<String>,
}

impl SelftestReport {
    /// Fold checks + the `not_validated` capability list into a report. See the
    /// module docs for the result rules.
    #[must_use]
    pub fn assemble(
        machine: Machine,
        checks: Vec<SelftestCheck>,
        not_validated: Vec<String>,
    ) -> Self {
        let scored = || checks.iter().filter(|c| is_scored(&c.name));
        let any_error = scored().any(|c| c.status == CheckStatus::Error);
        let any_fail = scored().any(|c| c.status == CheckStatus::Fail);
        let any_pass = scored().any(|c| c.status == CheckStatus::Pass);

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

    /// The report for a host with no CUDA device: a `device_present` error, every
    /// suite check skipped, and every capability unvalidated. `ERROR`, exit 2.
    #[must_use]
    pub fn no_device(full: bool) -> Self {
        let mut checks = vec![SelftestCheck::error(
            "device_present",
            "no CUDA device found; the oracle suite cannot run",
        )];
        checks.extend(
            CHECK_NAMES
                .iter()
                .filter(|n| full || !matches!(**n, "o5_cublas_gemm" | "vs_nsys"))
                .map(|n| SelftestCheck::skip(n, "no CUDA device")),
        );
        Self::assemble(
            Machine::default(),
            checks,
            NOT_VALIDATED_TOKENS
                .iter()
                .map(ToString::to_string)
                .collect(),
        )
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

    /// Structural self-consistency check. An empty list means the report is
    /// well-formed.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut problems = Vec::new();

        if self.schema_version != SCHEMA_VERSION {
            problems.push(format!(
                "schema_version {:?} != {SCHEMA_VERSION:?}",
                self.schema_version
            ));
        }
        if self.caliper_version.trim().is_empty() {
            problems.push("caliper_version is empty".to_string());
        }
        if self.checks.is_empty() {
            problems.push("report has no checks".to_string());
        }
        if self.checks.iter().any(|c| c.name.trim().is_empty()) {
            problems.push("a check has an empty name".to_string());
        }
        for token in &self.not_validated {
            if !NOT_VALIDATED_TOKENS.contains(&token.as_str()) {
                problems.push(format!("not_validated has an unknown token {token:?}"));
            }
        }

        let recomputed = Self::assemble(
            self.machine.clone(),
            self.checks.clone(),
            self.not_validated.clone(),
        );
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
        if self.result == Outcome::Pass
            && !self
                .checks
                .iter()
                .any(|c| is_scored(&c.name) && c.status == CheckStatus::Pass)
        {
            problems.push("result is PASS but no scored check passed".to_string());
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
    fn assemble(checks: Vec<SelftestCheck>) -> SelftestReport {
        SelftestReport::assemble(Machine::default(), checks, Vec::new())
    }

    #[test]
    fn all_pass_is_a_pass_exit_zero() {
        let r = assemble(vec![pass("o1_duration_linearity"), pass("o3_fma_peak")]);
        assert_eq!(r.result, Outcome::Pass);
        assert_eq!(r.coverage, Coverage::Reduced);
        assert_eq!(r.exit_code(), 0);
        assert!(r.not_validated.is_empty());
        assert!(r.validate().is_empty());
    }

    #[test]
    fn a_failing_scored_check_makes_the_result_fail() {
        let r = assemble(vec![pass("o1_duration_linearity"), fail("o3_fma_peak")]);
        assert_eq!(r.result, Outcome::Fail);
        assert_eq!(r.exit_code(), 1);
    }

    #[test]
    fn an_erroring_check_beats_a_failing_one() {
        let r = assemble(vec![
            fail("o1_duration_linearity"),
            SelftestCheck::error("o2_bandwidth", "boom"),
        ]);
        assert_eq!(r.result, Outcome::Error);
        assert_eq!(r.exit_code(), 2);
    }

    #[test]
    fn a_context_line_does_not_count_as_a_scored_pass() {
        // device_present PASS but every scored check skipped -> ERROR, not PASS.
        let r = assemble(vec![
            SelftestCheck::pass("device_present", "a CUDA device is present"),
            SelftestCheck::skip("o1_duration_linearity", "runner deferred"),
            SelftestCheck::skip("o3_fma_peak", "runner deferred"),
        ]);
        assert_eq!(r.result, Outcome::Error);
        assert_eq!(r.exit_code(), 2);
        assert!(r.validate().is_empty());
    }

    #[test]
    fn a_reduced_run_with_every_runnable_check_passing_is_still_a_pass() {
        let r = assemble(vec![
            pass("o1_duration_linearity"),
            pass("o3_fma_peak"),
            SelftestCheck::skip("vs_nsys", "nsys not on PATH"),
        ]);
        assert_eq!(r.result, Outcome::Pass);
        assert_eq!(r.coverage, Coverage::Reduced);
    }

    #[test]
    fn nsys_pass_lifts_coverage_to_full() {
        let r = assemble(vec![pass("o1_duration_linearity"), pass("vs_nsys")]);
        assert_eq!(r.coverage, Coverage::Full);
    }

    #[test]
    fn not_validated_only_accepts_the_capability_vocabulary() {
        let ok = SelftestReport::assemble(
            Machine::default(),
            vec![pass("o1_duration_linearity")],
            vec!["clock_lock".into(), "ncu_crosscheck".into()],
        );
        assert!(ok.validate().is_empty());

        let bad = SelftestReport::assemble(
            Machine::default(),
            vec![pass("o1_duration_linearity")],
            vec!["o2_bandwidth".into()],
        );
        assert!(bad.validate().iter().any(|p| p.contains("unknown token")));
    }

    #[test]
    fn no_device_report_is_error_exit_two() {
        let r = SelftestReport::no_device(true);
        assert_eq!(r.result, Outcome::Error);
        assert_eq!(r.exit_code(), 2);
        assert_eq!(r.coverage, Coverage::Reduced);
        assert_eq!(r.not_validated, NOT_VALIDATED_TOKENS);
        assert!(r.checks.iter().any(|c| c.name == "device_present"));
        assert!(r.validate().is_empty());

        // without --full the two extra checks are not listed
        let reduced = SelftestReport::no_device(false);
        assert!(!reduced.checks.iter().any(|c| c.name == "vs_nsys"));
    }

    #[test]
    fn empty_check_list_is_an_error_and_validate_flags_it() {
        let r = assemble(Vec::new());
        assert_eq!(r.result, Outcome::Error);
        assert!(r.validate().iter().any(|p| p.contains("no checks")));
    }

    #[test]
    fn report_round_trips_and_validate_catches_tampering() {
        let r = assemble(vec![pass("o1_duration_linearity"), pass("o2_bandwidth")]);
        let back: SelftestReport = serde_json::from_str(&r.to_json()).unwrap();
        assert_eq!(back, r);

        let mut bad_result = r.clone();
        bad_result.result = Outcome::Fail;
        assert!(!bad_result.validate().is_empty());

        let mut bad_cov = r.clone();
        bad_cov.coverage = Coverage::Full;
        assert!(!bad_cov.validate().is_empty());

        let mut fake_pass = r;
        fake_pass.checks = vec![SelftestCheck::skip("o1_duration_linearity", "x")];
        fake_pass.result = Outcome::Pass;
        assert!(fake_pass
            .validate()
            .iter()
            .any(|p| p.contains("no scored check passed")));
    }

    #[test]
    fn a_full_pass_report_is_producible_from_oracle_checks() {
        // The shape the on-device runner assembles.
        let oc = |name: &str, passed: bool| OracleCheck {
            name: name.to_string(),
            passed,
            measured: 1.0,
            expected: 1.0,
            tolerance: 0.03,
            detail: "d".to_string(),
        };
        let checks = vec![
            SelftestCheck::pass("device_present", "a CUDA device is present"),
            SelftestCheck::from_oracle("o1_duration_linearity", &oc("o1_linearity", true)),
            SelftestCheck::from_oracle("o2_bandwidth", &oc("o2_bandwidth", true)),
            SelftestCheck::from_oracle("o2_flush_ab", &oc("o2_flush_ab_large", true)),
            SelftestCheck::from_oracle("o3_fma_peak", &oc("o3_fma", true)),
            SelftestCheck::from_oracle("o4_launch_overhead", &oc("o4_launch_overhead", true)),
            SelftestCheck::from_oracle("o6_throttle", &oc("o6_throttle", true)),
            SelftestCheck::skip("o7_calibration_gemm", "no table entry for this SKU"),
            SelftestCheck::from_oracle("reproducibility", &oc("reproducibility", true)),
        ];
        let r = SelftestReport::assemble(
            Machine::default(),
            checks,
            vec![
                "ncu_crosscheck".into(),
                "clock_lock".into(),
                "powercap_throttle".into(),
            ],
        );
        assert_eq!(r.result, Outcome::Pass);
        assert_eq!(r.coverage, Coverage::Reduced);
        assert_eq!(r.exit_code(), 0);
        assert!(r.validate().is_empty());
        // the from_oracle line carries structured fields
        let o1 = r
            .checks
            .iter()
            .find(|c| c.name == "o1_duration_linearity")
            .unwrap();
        assert_eq!(o1.tolerance.as_deref(), Some("3.0%"));
        assert_eq!(o1.measured["value"], serde_json::json!(1.0));
    }
}
