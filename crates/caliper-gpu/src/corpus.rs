//! Resolving `corpus:*` bench targets.
//!
//! `caliper bench corpus:o1` names a built-in oracle kernel. The kernels live in
//! `kernels/` and run on a CUDA host; this module just maps the target name to
//! the kernel key the launcher uses and lists what's available.

/// The built-in oracle targets, as `(target, kernel_key, one-line description)`.
pub const ORACLE_TARGETS: &[(&str, &str, &str)] = &[
    (
        "corpus:o1",
        "oracle:busy",
        "calibrated-duration spin (pins the timing path)",
    ),
    (
        "corpus:o2",
        "oracle:triad",
        "streaming triad (pins GB/s and the L2 flush)",
    ),
    (
        "corpus:o3",
        "oracle:fma_peak",
        "register-resident FMA peak (pins TFLOP/s)",
    ),
    (
        "corpus:o4",
        "oracle:one_op",
        "single-instruction kernel (pins launch overhead)",
    ),
    (
        "corpus:o6",
        "oracle:throttle_bait",
        "sustained high-power FMA (pins throttle detection)",
    ),
];

/// The kernel key for a `corpus:*` target, or `None` if it isn't a known oracle.
#[must_use]
pub fn resolve(target: &str) -> Option<&'static str> {
    ORACLE_TARGETS
        .iter()
        .find(|(name, _, _)| *name == target)
        .map(|(_, key, _)| *key)
}

/// Whether `target` uses the `corpus:` namespace at all.
#[must_use]
pub fn is_corpus_target(target: &str) -> bool {
    target.starts_with("corpus:")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_known_targets_and_rejects_others() {
        assert_eq!(resolve("corpus:o1"), Some("oracle:busy"));
        assert_eq!(resolve("corpus:o6"), Some("oracle:throttle_bait"));
        assert_eq!(resolve("corpus:o9"), None);
        assert_eq!(resolve("mykernel.py::fn"), None);
        assert!(is_corpus_target("corpus:o1"));
        assert!(!is_corpus_target("mykernel.py::fn"));
    }

    #[test]
    fn every_target_row_is_well_formed() {
        for (name, key, desc) in ORACLE_TARGETS {
            assert!(name.starts_with("corpus:"));
            assert!(key.starts_with("oracle:"));
            assert!(!desc.is_empty());
        }
    }
}
