//! The machine fingerprint: the device/host identity recorded with every
//! result, plus a completeness check and toolchain-version parsing.
//!
//! The field set is [`crate::schema::Machine`] (Appendix B of the plan). A
//! fingerprint has to be *whole* for a result to be comparable across machines,
//! so [`check`] / [`assert_complete`] report any field that a real NVML snapshot
//! would have filled but this one did not.
//!
//! Fields split into two tiers:
//!
//! * **Required** -- the hardware and driver identity (`gpu_name`, `sm_arch`,
//!   memory sizes, `l2_bytes`, PCIe link, ECC/MIG/persistence, driver and CUDA
//!   versions, `nvml_version`, and the `ptxas` / `nvcc` toolchain). A missing
//!   one makes the fingerprint incomplete.
//! * **Recommended** -- framework versions (`triton`, `torch`) that are simply
//!   absent on a pure CUDA-C setup. Reported, but not an error.

use serde::de::Error as _;
use serde::Serialize;

use crate::schema::Machine;

/// Why a fingerprint field is expected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// Hardware / driver identity; a gap makes the fingerprint incomplete.
    Required,
    /// Framework context; absent on a CUDA-C-only host, never an error.
    Recommended,
}

/// One expected-but-absent fingerprint field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FieldGap {
    /// Dotted field path, e.g. `"toolkit.nvcc"`.
    pub field: &'static str,
    /// Whether the gap is an error or just a note.
    pub tier: Tier,
}

/// The result of [`check`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FingerprintCheck {
    /// `true` when there are no `Required` gaps.
    pub complete: bool,
    /// Required fields that are missing or blank.
    pub missing_required: Vec<&'static str>,
    /// Recommended fields that are missing or blank.
    pub missing_recommended: Vec<&'static str>,
}

fn blank(s: Option<&String>) -> bool {
    s.is_none_or(|v| v.trim().is_empty())
}

/// Every gap in `m`, both tiers, in field order.
#[must_use]
pub fn gaps(m: &Machine) -> Vec<FieldGap> {
    let mut out = Vec::new();
    let mut req = |field: &'static str, missing: bool| {
        if missing {
            out.push(FieldGap {
                field,
                tier: Tier::Required,
            });
        }
    };

    req("gpu_name", blank(m.gpu_name.as_ref()));
    req("sm_arch", blank(m.sm_arch.as_ref()));
    req("vram_mib", m.vram_mib.is_none());
    req("sm_count", m.sm_count.is_none());
    req("l2_bytes", m.l2_bytes.is_none());
    req("bar1_mib", m.bar1_mib.is_none());
    req("driver", blank(m.driver.as_ref()));
    req("cuda_runtime", blank(m.cuda_runtime.as_ref()));
    req("cuda_driver", blank(m.cuda_driver.as_ref()));
    req("nvml_version", blank(m.nvml_version.as_ref()));
    req("ecc", m.ecc.is_none());
    req("mig", blank(m.mig.as_ref()));
    req("persistence_mode", m.persistence_mode.is_none());
    req("pcie_gen", m.pcie_gen.is_none());
    req("pcie_width", m.pcie_width.is_none());
    req("toolkit.ptxas", blank(m.toolkit.ptxas.as_ref()));
    req("toolkit.nvcc", blank(m.toolkit.nvcc.as_ref()));

    if blank(m.toolkit.triton.as_ref()) {
        out.push(FieldGap {
            field: "toolkit.triton",
            tier: Tier::Recommended,
        });
    }
    if blank(m.toolkit.torch.as_ref()) {
        out.push(FieldGap {
            field: "toolkit.torch",
            tier: Tier::Recommended,
        });
    }

    out
}

/// Split [`gaps`] into a [`FingerprintCheck`].
#[must_use]
pub fn check(m: &Machine) -> FingerprintCheck {
    let mut missing_required = Vec::new();
    let mut missing_recommended = Vec::new();
    for g in gaps(m) {
        match g.tier {
            Tier::Required => missing_required.push(g.field),
            Tier::Recommended => missing_recommended.push(g.field),
        }
    }
    FingerprintCheck {
        complete: missing_required.is_empty(),
        missing_required,
        missing_recommended,
    }
}

/// Whether every `Required` field is present.
#[must_use]
pub fn is_complete(m: &Machine) -> bool {
    check(m).complete
}

/// `Ok` iff the fingerprint is complete; otherwise an error naming the missing
/// required fields.
///
/// # Errors
/// [`FingerprintError`] listing every missing `Required` field.
pub fn assert_complete(m: &Machine) -> Result<(), FingerprintError> {
    let missing: Vec<String> = check(m)
        .missing_required
        .into_iter()
        .map(str::to_string)
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(FingerprintError { missing })
    }
}

/// An incomplete fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FingerprintError {
    /// The missing required field paths.
    pub missing: Vec<String>,
}

impl std::fmt::Display for FingerprintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fingerprint is missing required field(s): {}",
            self.missing.join(", ")
        )
    }
}

impl std::error::Error for FingerprintError {}

/// Parse a [`Machine`] from JSON (lenient about *fields*, like the rest of the
/// schema: unknown keys are ignored, missing ones default). The document itself
/// must be a JSON object -- `serde` would otherwise accept a positional array.
///
/// # Errors
/// A `serde_json` error if `s` is not JSON, or is JSON but not an object.
pub fn from_json(s: &str) -> Result<Machine, serde_json::Error> {
    let value: serde_json::Value = serde_json::from_str(s)?;
    if !value.is_object() {
        return Err(serde_json::Error::custom(
            "fingerprint document must be a JSON object",
        ));
    }
    serde_json::from_value(value)
}

/// Serialise a [`Machine`] to canonical compact JSON.
///
/// # Panics
/// Never in practice: a [`Machine`] holds only JSON-representable values.
#[must_use]
pub fn to_json(m: &Machine) -> String {
    serde_json::to_string(m).expect("Machine always serialises")
}

// --- toolchain version parsing ---------------------------------------------

/// The CUDA toolchain version from `nvcc --version` output, e.g. `"12.4.131"`
/// (falling back to `"12.4"` when the build number is absent). `None` if the
/// text carries no recognisable version.
#[must_use]
pub fn parse_nvcc_version(output: &str) -> Option<String> {
    parse_cuda_tool_version(output)
}

/// The CUDA toolchain version from `ptxas --version` output. Same format as
/// [`parse_nvcc_version`].
#[must_use]
pub fn parse_ptxas_version(output: &str) -> Option<String> {
    parse_cuda_tool_version(output)
}

/// Both `nvcc --version` and `ptxas --version` print a
/// `Cuda compilation tools, release <maj.min>, V<maj.min.build>` line.
fn parse_cuda_tool_version(output: &str) -> Option<String> {
    let take_version = |s: &str| -> String {
        s.chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect::<String>()
    };

    let usable = |v: &str| v.contains('.') && !v.ends_with('.');

    // Prefer the fully-qualified `V12.4.131` token (also matches a 2-part
    // `V13.0` when a build number is not printed). `str::lines` already strips
    // a trailing `\r`, so CRLF output is handled.
    for line in output.lines() {
        if let Some(pos) = line.find(", V") {
            let v = take_version(&line[pos + 3..]);
            if usable(&v) {
                return Some(v);
            }
        }
    }
    // Fall back to `release 12.4`.
    for line in output.lines() {
        if let Some(pos) = line.find("release ") {
            let v = take_version(&line[pos + "release ".len()..]);
            let v = v.trim_end_matches('.').to_string();
            if usable(&v) {
                return Some(v);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Toolkit;

    fn full_machine() -> Machine {
        Machine {
            gpu_name: Some("NVIDIA GeForce RTX 4090".into()),
            sm_arch: Some("sm_89".into()),
            vram_mib: Some(24564),
            sm_count: Some(128),
            l2_bytes: Some(75_497_472),
            bar1_mib: Some(32768),
            driver: Some("550.90.07".into()),
            cuda_runtime: Some("12.4".into()),
            cuda_driver: Some("12.4".into()),
            nvml_version: Some("12.550.90".into()),
            ecc: Some(false),
            mig: Some("disabled".into()),
            persistence_mode: Some(true),
            pcie_gen: Some(4),
            pcie_width: Some(16),
            toolkit: Toolkit {
                triton: Some("3.2.0".into()),
                torch: Some("2.6.0".into()),
                ptxas: Some("12.4.131".into()),
                nvcc: Some("12.4.131".into()),
            },
        }
    }

    #[test]
    fn a_full_machine_is_complete_and_round_trips() {
        let m = full_machine();
        assert!(is_complete(&m));
        assert!(check(&m).missing_recommended.is_empty());
        assert!(assert_complete(&m).is_ok());
        assert_eq!(from_json(&to_json(&m)).unwrap(), m);
    }

    #[test]
    fn each_missing_required_field_is_reported() {
        let base = full_machine();

        let mut m = base.clone();
        m.sm_arch = None;
        assert!(!is_complete(&m));
        assert_eq!(check(&m).missing_required, vec!["sm_arch"]);
        let err = assert_complete(&m).unwrap_err();
        assert!(err.to_string().contains("sm_arch"));

        let mut m = base.clone();
        m.l2_bytes = None;
        m.pcie_gen = None;
        m.toolkit.nvcc = None;
        let c = check(&m);
        assert!(!c.complete);
        assert_eq!(
            c.missing_required,
            vec!["l2_bytes", "pcie_gen", "toolkit.nvcc"]
        );

        // a blank string counts as missing, not present
        let mut m = base;
        m.driver = Some("   ".into());
        assert_eq!(check(&m).missing_required, vec!["driver"]);
    }

    #[test]
    fn framework_versions_are_recommended_not_required() {
        let mut m = full_machine();
        m.toolkit.triton = None;
        m.toolkit.torch = None;
        let c = check(&m);
        assert!(c.complete); // still complete
        assert_eq!(
            c.missing_recommended,
            vec!["toolkit.triton", "toolkit.torch"]
        );
    }

    #[test]
    fn an_empty_machine_lists_every_required_field() {
        let c = check(&Machine::default());
        assert!(!c.complete);
        // Canary: bump this (and the Python mirror) when a field is added to
        // the required set. Every reported field is distinct.
        assert_eq!(c.missing_required.len(), 17);
        let mut uniq = c.missing_required.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 17);
        assert!(c.missing_recommended.contains(&"toolkit.triton"));
    }

    #[test]
    fn gaps_returns_both_tiers_in_field_order() {
        let g = gaps(&Machine::default());
        assert_eq!(g.first().unwrap().field, "gpu_name");
        assert!(g
            .iter()
            .any(|x| x.field == "toolkit.nvcc" && x.tier == Tier::Required));
        assert!(g
            .iter()
            .any(|x| x.field == "toolkit.torch" && x.tier == Tier::Recommended));
        // required fields come before recommended ones
        let last_req = g.iter().rposition(|x| x.tier == Tier::Required).unwrap();
        let first_rec = g.iter().position(|x| x.tier == Tier::Recommended).unwrap();
        assert!(last_req < first_rec);
    }

    #[test]
    fn from_json_rejects_non_object_documents() {
        assert!(from_json("[]").is_err());
        assert!(from_json("42").is_err());
        assert!(from_json("\"sm_89\"").is_err());
    }

    #[test]
    fn nvcc_version_is_parsed_from_the_tools_line() {
        let out = "nvcc: NVIDIA (R) Cuda compiler driver\n\
                   Copyright (c) 2005-2024 NVIDIA Corporation\n\
                   Built on Tue_Feb_27_16:19:38_PST_2024\n\
                   Cuda compilation tools, release 12.4, V12.4.131\n\
                   Build cuda_12.4.r12.4/compiler.34097967_0\n";
        assert_eq!(parse_nvcc_version(out).as_deref(), Some("12.4.131"));
    }

    #[test]
    fn ptxas_version_has_the_same_shape() {
        let out = "ptxas: NVIDIA (R) Ptx optimizing assembler\n\
                   Copyright (c) 2005-2025 NVIDIA Corporation\n\
                   Cuda compilation tools, release 12.6, V12.6.20\n";
        assert_eq!(parse_ptxas_version(out).as_deref(), Some("12.6.20"));
    }

    #[test]
    fn version_falls_back_to_the_release_number() {
        let out = "Cuda compilation tools, release 11.8, and nothing else\n";
        assert_eq!(parse_nvcc_version(out).as_deref(), Some("11.8"));
    }

    #[test]
    fn version_handles_crlf_cuda13_and_a_missing_build_number() {
        // CRLF line endings (str::lines strips the trailing \r).
        let crlf = "nvcc: NVIDIA (R) Cuda compiler driver\r\n\
                    Cuda compilation tools, release 12.4, V12.4.131\r\n";
        assert_eq!(parse_nvcc_version(crlf).as_deref(), Some("12.4.131"));

        // CUDA 13.
        let c13 = "Cuda compilation tools, release 13.0, V13.0.48\n";
        assert_eq!(parse_nvcc_version(c13).as_deref(), Some("13.0.48"));

        // The V token with no build number is still preferred over `release`.
        let two = "Cuda compilation tools, release 12.9, V12.9\n";
        assert_eq!(parse_nvcc_version(two).as_deref(), Some("12.9"));
    }

    #[test]
    fn unrecognised_text_yields_none() {
        assert_eq!(parse_nvcc_version(""), None);
        assert_eq!(parse_nvcc_version("command not found"), None);
        assert_eq!(parse_ptxas_version("release without a number"), None);
        // a ", V" that is not a version token
        assert_eq!(
            parse_nvcc_version("Redistribution, Verification prohibited"),
            None
        );
    }
}
