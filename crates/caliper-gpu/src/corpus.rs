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

/// Non-oracle reference kernels in the `corpus:` namespace -- workloads with a
/// known FLOP / byte count that the roofline model can be checked against.
pub const REFERENCE_TARGETS: &[(&str, &str, &str)] = &[
    (
        "corpus:gemm",
        "corpus:gemm_bf16",
        "dense bf16 GEMM (roofline / cuBLAS reference)",
    ),
    (
        "corpus:rmsnorm",
        "corpus:rmsnorm",
        "RMSNorm forward (roofline / torch reference)",
    ),
    (
        "corpus:softmax",
        "corpus:softmax",
        "row-wise softmax forward (roofline / torch reference)",
    ),
    (
        "corpus:attention_fwd",
        "corpus:attention_fwd",
        "FlashAttention-style forward (roofline / SDPA reference)",
    ),
    (
        "corpus:attention_bwd",
        "corpus:attention_bwd",
        "FlashAttention-style backward (roofline / SDPA-backward reference)",
    ),
];

/// Every `corpus:*` target: the oracles plus the reference kernels.
#[must_use]
pub fn all_targets() -> Vec<(&'static str, &'static str, &'static str)> {
    ORACLE_TARGETS
        .iter()
        .chain(REFERENCE_TARGETS)
        .copied()
        .collect()
}

/// The kernel key for a `corpus:*` target, or `None` if it isn't a known one.
#[must_use]
pub fn resolve(target: &str) -> Option<&'static str> {
    ORACLE_TARGETS
        .iter()
        .chain(REFERENCE_TARGETS)
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
        assert_eq!(resolve("corpus:gemm"), Some("corpus:gemm_bf16"));
        assert_eq!(resolve("corpus:rmsnorm"), Some("corpus:rmsnorm"));
        assert_eq!(resolve("corpus:softmax"), Some("corpus:softmax"));
        assert_eq!(
            resolve("corpus:attention_fwd"),
            Some("corpus:attention_fwd")
        );
        assert_eq!(
            resolve("corpus:attention_bwd"),
            Some("corpus:attention_bwd")
        );
        assert_eq!(resolve("corpus:o9"), None);
        assert_eq!(resolve("mykernel.py::fn"), None);
        assert!(is_corpus_target("corpus:o1"));
        assert!(!is_corpus_target("mykernel.py::fn"));
    }

    #[test]
    fn every_target_row_is_well_formed() {
        for (name, key, desc) in all_targets() {
            assert!(name.starts_with("corpus:"));
            assert!(!key.is_empty());
            assert!(!desc.is_empty());
        }
        for (_, key, _) in ORACLE_TARGETS {
            assert!(key.starts_with("oracle:"));
        }
        assert!(all_targets().len() > ORACLE_TARGETS.len());
    }
}
