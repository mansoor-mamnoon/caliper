//! "Is this machine fit to benchmark?" -- the assessment behind `caliper doctor`.
//!
//! [`assess`] is pure: it takes a bundle of device facts ([`DoctorFacts`],
//! gathered by `caliper-gpu`) and returns a [`DoctorReport`] with a verdict, an
//! environment classification, per-check detail, and an exit code. Nothing here
//! touches hardware, so every branch is `cargo test`-covered.
//!
//! The verdict is deliberately lenient: only an active throttle or a missing
//! device blocks. A denied clock lock, restricted counters, ECC, MIG, or a
//! non-persistent driver *reduce confidence* (`environment: Constrained`) but
//! still leave the machine "fit" -- caliper tags such runs and stays useful for
//! relative comparisons.

use serde::{Deserialize, Serialize};

/// Everything [`assess`] needs. `caliper-gpu` fills this from the device layer.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DoctorFacts {
    /// Whether a device was found at all.
    pub device_found: bool,
    /// `Some(true)` if a lock probe succeeded, `Some(false)` if denied,
    /// `None` if not probed.
    pub clocks_lockable: Option<bool>,
    /// Throttle reasons active right now (NVML names).
    pub active_throttle: Vec<String>,
    /// Whether ECC is enabled.
    pub ecc_enabled: Option<bool>,
    /// MIG state: `"disabled"` / `None` is fine, anything else is a partition.
    pub mig: Option<String>,
    /// Whether persistence mode is on.
    pub persistence_mode: Option<bool>,
    /// MiB of device memory in use by *other* processes.
    pub background_load_mib: Option<u64>,
    /// Whether performance counters (ncu-class) are available.
    pub counters_available: Option<bool>,
}

/// Overall fitness verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// Fit to benchmark (possibly with reduced confidence -- see `environment`).
    Fit,
    /// Not fit: something is actively wrong (throttling).
    Unfit,
    /// Could not assess: no device.
    Error,
}

/// Environment classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Environment {
    /// Clocks lockable, counters available -- absolute numbers are trustworthy.
    Normal,
    /// Some capability is missing (Colab-like); results are tagged and good for
    /// relative comparison only.
    Constrained,
}

/// Status of one check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    /// Nothing wrong.
    Pass,
    /// A caveat that reduces confidence but does not block.
    Warn,
    /// Blocks benchmarking.
    Fail,
    /// Not applicable / not probed.
    Skip,
}

/// One check line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DoctorCheck {
    /// Short check name.
    pub name: String,
    /// Its status.
    pub status: CheckStatus,
    /// Human-readable detail.
    pub detail: String,
}

/// The full report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DoctorReport {
    /// Overall verdict.
    pub verdict: Verdict,
    /// Environment classification.
    pub environment: Environment,
    /// Every check that ran.
    pub checks: Vec<DoctorCheck>,
    /// Notes explaining any reduced confidence.
    pub notes: Vec<String>,
    /// Process exit code: 0 fit, 1 unfit, 2 error.
    pub exit_code: i32,
}

impl DoctorReport {
    /// A report for "no device found".
    #[must_use]
    pub fn no_device() -> Self {
        Self {
            verdict: Verdict::Error,
            environment: Environment::Constrained,
            checks: vec![DoctorCheck {
                name: "device".to_string(),
                status: CheckStatus::Fail,
                detail: "no CUDA device found -- run caliper on a host with a GPU".to_string(),
            }],
            notes: vec![],
            exit_code: 2,
        }
    }

    /// Serialise to canonical JSON.
    ///
    /// # Panics
    /// Never in practice; the report holds only JSON-representable values.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("DoctorReport always serialises")
    }

    /// Render for a terminal.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::from("caliper doctor\n");
        let verdict = match self.verdict {
            Verdict::Fit if self.environment == Environment::Constrained => {
                "FIT TO BENCHMARK (reduced confidence)"
            }
            Verdict::Fit => "FIT TO BENCHMARK",
            Verdict::Unfit => "NOT FIT",
            Verdict::Error => "CANNOT ASSESS",
        };
        out.push_str(&format!("  verdict:     {verdict}\n"));
        out.push_str(&format!(
            "  environment: {}\n",
            match self.environment {
                Environment::Normal => "normal",
                Environment::Constrained => "constrained (Colab-like)",
            }
        ));
        for c in &self.checks {
            let tag = match c.status {
                CheckStatus::Pass => "PASS",
                CheckStatus::Warn => "WARN",
                CheckStatus::Fail => "FAIL",
                CheckStatus::Skip => "SKIP",
            };
            out.push_str(&format!("  [{tag}] {} -- {}\n", c.name, c.detail));
        }
        for n in &self.notes {
            out.push_str(&format!("  note: {n}\n"));
        }
        out
    }
}

/// Assess a machine from gathered facts.
#[must_use]
pub fn assess(facts: &DoctorFacts) -> DoctorReport {
    if !facts.device_found {
        return DoctorReport::no_device();
    }

    let mut checks = Vec::new();
    let mut notes = Vec::new();
    let mut constrained = false;

    // clock locking
    checks.push(match facts.clocks_lockable {
        Some(true) => check("clock lock", CheckStatus::Pass, "clocks are lockable"),
        Some(false) => {
            constrained = true;
            notes.push(
                "GPU clock locking denied -> results tagged `clocks-unlocked`, reduced confidence"
                    .to_string(),
            );
            check(
                "clock lock",
                CheckStatus::Warn,
                "denied by the driver (fine for relative comparisons on this box)",
            )
        }
        None => check("clock lock", CheckStatus::Skip, "not probed"),
    });

    // active throttle -- the one hard blocker
    if facts.active_throttle.is_empty() {
        checks.push(check(
            "throttling",
            CheckStatus::Pass,
            "not throttling right now",
        ));
    } else {
        checks.push(check(
            "throttling",
            CheckStatus::Fail,
            &format!(
                "GPU is throttling right now: {} -- measurements will be wrong",
                facts.active_throttle.join(", ")
            ),
        ));
    }

    // ECC
    checks.push(match facts.ecc_enabled {
        Some(true) => check(
            "ecc",
            CheckStatus::Warn,
            "ECC on (small bandwidth/latency cost)",
        ),
        Some(false) => check("ecc", CheckStatus::Pass, "ECC off"),
        None => check("ecc", CheckStatus::Skip, "unknown"),
    });

    // MIG
    checks.push(match facts.mig.as_deref() {
        None | Some("disabled") => check("mig", CheckStatus::Pass, "not partitioned"),
        Some(geom) => check(
            "mig",
            CheckStatus::Warn,
            &format!("MIG partition active ({geom}); measuring a slice, not the device"),
        ),
    });

    // persistence mode
    checks.push(match facts.persistence_mode {
        Some(true) => check("persistence mode", CheckStatus::Pass, "on"),
        Some(false) => check(
            "persistence mode",
            CheckStatus::Warn,
            "off (adds per-run init latency and clock instability)",
        ),
        None => check("persistence mode", CheckStatus::Skip, "unknown"),
    });

    // background load
    checks.push(match facts.background_load_mib {
        Some(mib) if mib > 256 => check(
            "background load",
            CheckStatus::Warn,
            &format!("another process is using {mib} MiB -- close it for clean numbers"),
        ),
        Some(mib) => check(
            "background load",
            CheckStatus::Pass,
            &format!("{mib} MiB in use elsewhere"),
        ),
        None => check("background load", CheckStatus::Skip, "not measured"),
    });

    // performance counters
    checks.push(match facts.counters_available {
        Some(true) => check("performance counters", CheckStatus::Pass, "available"),
        Some(false) => {
            constrained = true;
            notes.push(
                "performance counters restricted -> ncu-class metrics unavailable; using ptxas + occupancy API"
                    .to_string(),
            );
            check(
                "performance counters",
                CheckStatus::Warn,
                "restricted (typical on shared/cloud hosts)",
            )
        }
        None => check("performance counters", CheckStatus::Skip, "not probed"),
    });

    let has_fail = checks.iter().any(|c| c.status == CheckStatus::Fail);
    let verdict = if has_fail {
        Verdict::Unfit
    } else {
        Verdict::Fit
    };
    let environment = if constrained {
        Environment::Constrained
    } else {
        Environment::Normal
    };
    let exit_code = match verdict {
        Verdict::Fit => 0,
        Verdict::Unfit => 1,
        Verdict::Error => 2,
    };

    DoctorReport {
        verdict,
        environment,
        checks,
        notes,
        exit_code,
    }
}

fn check(name: &str, status: CheckStatus, detail: &str) -> DoctorCheck {
    DoctorCheck {
        name: name.to_string(),
        status,
        detail: detail.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy() -> DoctorFacts {
        DoctorFacts {
            device_found: true,
            clocks_lockable: Some(true),
            active_throttle: vec![],
            ecc_enabled: Some(false),
            mig: Some("disabled".to_string()),
            persistence_mode: Some(true),
            background_load_mib: Some(0),
            counters_available: Some(true),
        }
    }

    #[test]
    fn a_healthy_box_is_fit_normal_exit_zero() {
        let r = assess(&healthy());
        assert_eq!(r.verdict, Verdict::Fit);
        assert_eq!(r.environment, Environment::Normal);
        assert_eq!(r.exit_code, 0);
        assert!(r.checks.iter().all(|c| c.status == CheckStatus::Pass));
        assert!(r.notes.is_empty());
    }

    #[test]
    fn no_device_is_error_exit_two() {
        let r = assess(&DoctorFacts::default());
        assert_eq!(r.verdict, Verdict::Error);
        assert_eq!(r.exit_code, 2);
        assert_eq!(r.checks[0].status, CheckStatus::Fail);
    }

    #[test]
    fn active_throttle_is_unfit_exit_one() {
        let mut f = healthy();
        f.active_throttle = vec!["SW_THERMAL_SLOWDOWN".to_string()];
        let r = assess(&f);
        assert_eq!(r.verdict, Verdict::Unfit);
        assert_eq!(r.exit_code, 1);
        let t = r.checks.iter().find(|c| c.name == "throttling").unwrap();
        assert_eq!(t.status, CheckStatus::Fail);
    }

    #[test]
    fn denied_lock_and_restricted_counters_are_constrained_but_still_fit() {
        let mut f = healthy();
        f.clocks_lockable = Some(false);
        f.counters_available = Some(false);
        let r = assess(&f);
        assert_eq!(r.verdict, Verdict::Fit);
        assert_eq!(r.environment, Environment::Constrained);
        assert_eq!(r.exit_code, 0);
        assert_eq!(r.notes.len(), 2);
        assert!(r.render().contains("reduced confidence"));
        assert!(r.render().contains("constrained"));
    }

    #[test]
    fn ecc_mig_persistence_background_are_warnings_not_blockers() {
        let mut f = healthy();
        f.ecc_enabled = Some(true);
        f.mig = Some("1g.10gb".to_string());
        f.persistence_mode = Some(false);
        f.background_load_mib = Some(4096);
        let r = assess(&f);
        assert_eq!(r.verdict, Verdict::Fit);
        assert_eq!(r.environment, Environment::Normal); // warns, but not a "constrained" trigger
        assert_eq!(r.exit_code, 0);
        let warns = r
            .checks
            .iter()
            .filter(|c| c.status == CheckStatus::Warn)
            .count();
        assert_eq!(warns, 4);
    }

    #[test]
    fn report_json_round_trips() {
        let r = assess(&healthy());
        let back: DoctorReport = serde_json::from_str(&r.to_json()).unwrap();
        assert_eq!(back, r);
    }
}
