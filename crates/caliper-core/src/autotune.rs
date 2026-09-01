//! The autotune-per-config timing cache key.
//!
//! A kernel's autotune configs are timed separately, and a config's time is
//! reusable only on the *same* environment: same SKU, driver, CUDA, compiler,
//! frameworks, and the same kernel source. [`AutotuneKey`] captures exactly
//! that, plus the config itself; [`AutotuneKey::to_key`] renders it to a stable
//! string a cache keys on. Adding a config to a kernel changes only that
//! config's key, so a re-sweep re-times only the new one.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::schema::Machine;

/// The identity of one (environment, kernel, config) triple.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutotuneKey {
    /// Compute-capability tag, e.g. `"sm_89"`.
    pub sku: String,
    /// NVIDIA driver version.
    pub driver: String,
    /// CUDA runtime version.
    pub cuda: String,
    /// `ptxas` version.
    pub ptxas: String,
    /// Triton version (`""` if not installed).
    pub triton: String,
    /// PyTorch version (`""` if not installed).
    pub torch: String,
    /// Content hash of the kernel source.
    pub kernel_source_hash: String,
    /// Canonical JSON of the autotune config (object keys sorted).
    pub config_canonical: String,
}

impl AutotuneKey {
    /// Build a key from a machine fingerprint, the kernel source hash, and the
    /// config as JSON.
    ///
    /// # Errors
    /// Returns the `serde_json` error if `config_json` is not valid JSON.
    pub fn build(
        machine: &Machine,
        kernel_source_hash: &str,
        config_json: &str,
    ) -> Result<Self, serde_json::Error> {
        let s = |o: &Option<String>| o.clone().unwrap_or_default();
        Ok(Self {
            sku: s(&machine.sm_arch),
            driver: s(&machine.driver),
            cuda: s(&machine.cuda_runtime),
            ptxas: s(&machine.toolkit.ptxas),
            triton: s(&machine.toolkit.triton),
            torch: s(&machine.toolkit.torch),
            kernel_source_hash: kernel_source_hash.to_string(),
            config_canonical: canonical_json(config_json)?,
        })
    }

    /// A stable, human-readable cache key. Order-independent in the config
    /// (keys are sorted) and stable across builds.
    #[must_use]
    pub fn to_key(&self) -> String {
        format!(
            "sku={}|driver={}|cuda={}|ptxas={}|triton={}|torch={}|ksh={}|config={}",
            self.sku,
            self.driver,
            self.cuda,
            self.ptxas,
            self.triton,
            self.torch,
            self.kernel_source_hash,
            self.config_canonical
        )
    }
}

/// Re-serialise a JSON document with object keys sorted and no whitespace --
/// `serde_json`'s `Map` is a `BTreeMap`, so `to_string` after `from_str` is
/// canonical.
///
/// # Errors
/// The `serde_json` error if `s` is not valid JSON.
pub fn canonical_json(s: &str) -> Result<String, serde_json::Error> {
    let value: Value = serde_json::from_str(s)?;
    serde_json::to_string(&value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Machine, Toolkit};

    fn machine() -> Machine {
        Machine {
            sm_arch: Some("sm_89".into()),
            driver: Some("550.90.07".into()),
            cuda_runtime: Some("12.4".into()),
            toolkit: Toolkit {
                ptxas: Some("12.4.131".into()),
                triton: Some("3.2.0".into()),
                torch: Some("2.6.0".into()),
                nvcc: None,
            },
            ..Machine::default()
        }
    }

    #[test]
    fn config_key_is_order_independent() {
        let a =
            AutotuneKey::build(&machine(), "src:abc", r#"{"BLOCK_M":128,"num_warps":8}"#).unwrap();
        let b =
            AutotuneKey::build(&machine(), "src:abc", r#"{"num_warps":8,"BLOCK_M":128}"#).unwrap();
        assert_eq!(a.to_key(), b.to_key());
        assert_eq!(a.config_canonical, r#"{"BLOCK_M":128,"num_warps":8}"#);
    }

    #[test]
    fn changing_any_component_changes_the_key() {
        let base = AutotuneKey::build(&machine(), "src:abc", r#"{"BLOCK_M":128}"#).unwrap();

        let other_config = AutotuneKey::build(&machine(), "src:abc", r#"{"BLOCK_M":256}"#).unwrap();
        assert_ne!(base.to_key(), other_config.to_key());

        let other_src = AutotuneKey::build(&machine(), "src:def", r#"{"BLOCK_M":128}"#).unwrap();
        assert_ne!(base.to_key(), other_src.to_key());

        let mut m = machine();
        m.driver = Some("560.0.0".into());
        let other_driver = AutotuneKey::build(&m, "src:abc", r#"{"BLOCK_M":128}"#).unwrap();
        assert_ne!(base.to_key(), other_driver.to_key());
    }

    #[test]
    fn a_missing_framework_is_an_empty_component_not_an_error() {
        let mut m = machine();
        m.toolkit.triton = None;
        m.toolkit.torch = None;
        let k = AutotuneKey::build(&m, "src:abc", "{}").unwrap();
        assert!(k.to_key().contains("|triton=|"));
        assert_eq!(k.config_canonical, "{}");
    }

    #[test]
    fn build_rejects_a_non_json_config() {
        assert!(AutotuneKey::build(&machine(), "src:abc", "{not json").is_err());
    }

    #[test]
    fn canonical_json_sorts_nested_objects() {
        assert_eq!(
            canonical_json(r#"{"b":1,"a":{"y":2,"x":1}}"#).unwrap(),
            r#"{"a":{"x":1,"y":2},"b":1}"#
        );
    }
}
