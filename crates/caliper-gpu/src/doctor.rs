//! Gather [`DoctorFacts`] from a device layer and assess them.
//!
//! `caliper-core` owns the pure [`caliper_core::doctor::assess`] logic; this
//! module is the thin part that probes a live (or replayed) device.

use caliper_core::doctor::{assess, DoctorFacts, DoctorReport};

use crate::ports::{DeviceInfo, GpuClock};
use crate::types::{ClockTarget, LockOutcome};

/// Probe `layer` for the facts `assess` needs.
///
/// The sequence is `snapshot`, a lock probe (`lock` then `unlock` if it took),
/// and a throttle read. If `snapshot` fails, `device_found` is `false` and the
/// rest is left unprobed.
pub fn gather<L: DeviceInfo + GpuClock + ?Sized>(layer: &mut L) -> DoctorFacts {
    let machine = match layer.snapshot() {
        Ok(m) => m,
        Err(_) => return DoctorFacts::default(), // device_found = false
    };

    let clocks_lockable = match layer.lock(ClockTarget::default()) {
        Ok(LockOutcome::Locked) => {
            let _ = layer.unlock();
            Some(true)
        }
        Ok(LockOutcome::Denied | LockOutcome::Unsupported) => Some(false),
        Err(_) => Some(false),
    };

    let active_throttle = layer.throttle_reasons().unwrap_or_default();

    DoctorFacts {
        device_found: true,
        clocks_lockable,
        active_throttle,
        ecc_enabled: machine.ecc,
        mig: machine.mig,
        persistence_mode: machine.persistence_mode,
        background_load_mib: None, // needs the NVML process list; filled on-device
        counters_available: None,  // needs a counter probe; filled on-device
    }
}

/// Gather and assess in one step.
pub fn run<L: DeviceInfo + GpuClock + ?Sized>(layer: &mut L) -> DoctorReport {
    assess(&gather(layer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::FixturePlayer;
    use caliper_core::doctor::{CheckStatus, Environment, Verdict};

    fn player(jsonl: &str) -> FixturePlayer {
        FixturePlayer::from_jsonl(jsonl).unwrap()
    }

    const M: &str = r#"{"ecc":false,"mig":"disabled","persistence_mode":true,"sm_arch":"sm_89"}"#;

    #[test]
    fn a_healthy_replayed_device_is_fit() {
        let jsonl = format!(
            concat!(
                r#"{{"port":"device_info","method":"snapshot","args":null,"ret":{{"Ok":{m}}}}}"#,
                "\n",
                r#"{{"port":"gpu_clock","method":"lock","args":{{"sm_mhz":null,"mem_mhz":null}},"ret":{{"Ok":"Locked"}}}}"#,
                "\n",
                r#"{{"port":"gpu_clock","method":"unlock","args":null,"ret":{{"Ok":null}}}}"#,
                "\n",
                r#"{{"port":"gpu_clock","method":"throttle_reasons","args":null,"ret":{{"Ok":[]}}}}"#,
            ),
            m = M
        );
        let mut p = player(&jsonl);
        let r = run(&mut p);
        assert_eq!(r.verdict, Verdict::Fit);
        assert_eq!(r.environment, Environment::Normal);
        assert_eq!(r.exit_code, 0);
    }

    #[test]
    fn a_denied_lock_is_constrained_but_fit() {
        let jsonl = format!(
            concat!(
                r#"{{"port":"device_info","method":"snapshot","args":null,"ret":{{"Ok":{m}}}}}"#,
                "\n",
                r#"{{"port":"gpu_clock","method":"lock","args":{{"sm_mhz":null,"mem_mhz":null}},"ret":{{"Ok":"Denied"}}}}"#,
                "\n",
                r#"{{"port":"gpu_clock","method":"throttle_reasons","args":null,"ret":{{"Ok":[]}}}}"#,
            ),
            m = M
        );
        let mut p = player(&jsonl);
        let r = run(&mut p);
        assert_eq!(r.verdict, Verdict::Fit);
        assert_eq!(r.environment, Environment::Constrained);
        let lock = r.checks.iter().find(|c| c.name == "clock lock").unwrap();
        assert_eq!(lock.status, CheckStatus::Warn);
    }

    #[test]
    fn a_throttling_device_is_unfit() {
        let jsonl = format!(
            concat!(
                r#"{{"port":"device_info","method":"snapshot","args":null,"ret":{{"Ok":{m}}}}}"#,
                "\n",
                r#"{{"port":"gpu_clock","method":"lock","args":{{"sm_mhz":null,"mem_mhz":null}},"ret":{{"Ok":"Locked"}}}}"#,
                "\n",
                r#"{{"port":"gpu_clock","method":"unlock","args":null,"ret":{{"Ok":null}}}}"#,
                "\n",
                r#"{{"port":"gpu_clock","method":"throttle_reasons","args":null,"ret":{{"Ok":["SW_THERMAL_SLOWDOWN"]}}}}"#,
            ),
            m = M
        );
        let mut p = player(&jsonl);
        let r = run(&mut p);
        assert_eq!(r.verdict, Verdict::Unfit);
        assert_eq!(r.exit_code, 1);
    }

    #[test]
    fn a_snapshot_error_means_no_device() {
        let jsonl =
            r#"{"port":"device_info","method":"snapshot","args":null,"ret":{"Err":"NoDevice"}}"#;
        let mut p = player(jsonl);
        let r = run(&mut p);
        assert_eq!(r.verdict, Verdict::Error);
        assert_eq!(r.exit_code, 2);
    }
}
