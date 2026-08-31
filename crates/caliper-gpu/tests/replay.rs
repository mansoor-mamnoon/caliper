//! L1: replay recorded device sessions with no GPU, and check that a recording
//! made through `Recorder` replays identically.

use std::path::PathBuf;

use caliper_gpu::error::GpuError;
use caliper_gpu::fixture::{FixturePlayer, Recorder};
use caliper_gpu::ports::{DeviceInfo, GpuClock, KernelLauncher};
use caliper_gpu::types::{ClockTarget, LaunchSpec, LockOutcome};

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(rel)
}

fn player(rel: &str) -> FixturePlayer {
    FixturePlayer::from_path(fixture(rel)).expect("fixture loads")
}

#[test]
fn device_info_fixture_yields_the_recorded_machine() {
    let mut p = player("device_info/rtx4090.jsonl");
    let m = p.snapshot().unwrap();
    assert_eq!(m.sm_arch.as_deref(), Some("sm_89"));
    assert_eq!(m.gpu_name.as_deref(), Some("NVIDIA GeForce RTX 4090"));
    assert_eq!(m.l2_bytes, Some(75_497_472));
    assert_eq!(m.sm_count, Some(128));
    assert_eq!(m.pcie_gen, Some(4));
    assert_eq!(m.ecc, Some(false));
    assert_eq!(m.toolkit.triton.as_deref(), Some("3.2.0"));
    assert_eq!(p.remaining(), 0);
}

#[test]
fn clock_lock_ok_flow_replays_in_order() {
    let mut p = player("gpu_clock/lock_ok.jsonl");

    let outcome = p
        .lock(ClockTarget {
            sm_mhz: Some(2520),
            mem_mhz: None,
        })
        .unwrap();
    assert_eq!(outcome, LockOutcome::Locked);

    let state = p.read().unwrap();
    assert!(state.locked);
    assert_eq!(state.sm_mhz, Some(2520));
    assert_eq!(state.lock_method.as_deref(), Some("nvml"));

    assert!(p.throttle_reasons().unwrap().is_empty());
    p.unlock().unwrap();
    assert_eq!(p.remaining(), 0);

    // ClockState converts cleanly into the schema type.
    let clocks: caliper_core::schema::Clocks = state.into();
    assert_eq!(clocks.locked, Some(true));
}

#[test]
fn clock_lock_denied_is_reported_as_denied_not_an_error() {
    let mut p = player("gpu_clock/lock_denied.jsonl");
    assert_eq!(
        p.lock(ClockTarget {
            sm_mhz: Some(2520),
            mem_mhz: None
        })
        .unwrap(),
        LockOutcome::Denied
    );
    let state = p.read().unwrap();
    assert!(!state.locked);
    assert_eq!(
        p.throttle_reasons().unwrap(),
        vec![
            "SW_POWER_CAP".to_string(),
            "SW_THERMAL_SLOWDOWN".to_string()
        ]
    );
}

#[test]
fn clock_lock_hard_error_surfaces_as_permission_denied() {
    let mut p = player("gpu_clock/lock_error.jsonl");
    let err = p
        .lock(ClockTarget {
            sm_mhz: Some(2520),
            mem_mhz: None,
        })
        .unwrap_err();
    assert!(matches!(err, GpuError::PermissionDenied(_)), "got {err:?}");
}

#[test]
fn kernel_launcher_fixture_returns_the_recorded_batches() {
    let mut p = player("kernel_launcher/busy_200us.jsonl");
    let spec = LaunchSpec {
        kernel_key: "busy:200000ns".to_string(),
        batch: 32,
        batches: 10,
        use_graph: false,
    };
    let raw = p.time_batches(&spec).unwrap();
    assert_eq!(raw.gpu_us.len(), 10);
    assert_eq!(raw.batch, 32);
    assert!(raw.gpu_us.iter().all(|&x| (6000.0..7000.0).contains(&x)));
    assert!(raw.wall_us.iter().zip(&raw.gpu_us).all(|(w, g)| w > g)); // wall includes launch overhead
}

#[test]
fn arg_mismatch_is_caught() {
    let mut p = player("gpu_clock/lock_ok.jsonl");
    // recording expects sm_mhz=2520; ask for something else
    let err = p
        .lock(ClockTarget {
            sm_mhz: Some(1800),
            mem_mhz: None,
        })
        .unwrap_err();
    assert!(matches!(err, GpuError::FixtureMismatch { .. }));
}

#[test]
fn record_through_a_player_then_replay_is_identical() {
    let tmp = tempfile::NamedTempFile::new().unwrap();

    // A FixturePlayer stands in for a "real" port while we record it.
    let inner = player("gpu_clock/lock_ok.jsonl");
    let mut rec = Recorder::new(inner, tmp.path()).unwrap();

    let o1 = rec
        .lock(ClockTarget {
            sm_mhz: Some(2520),
            mem_mhz: None,
        })
        .unwrap();
    let s1 = rec.read().unwrap();
    let t1 = rec.throttle_reasons().unwrap();
    rec.unlock().unwrap();
    drop(rec); // flush

    let mut replay = FixturePlayer::from_path(tmp.path()).unwrap();
    assert_eq!(
        replay
            .lock(ClockTarget {
                sm_mhz: Some(2520),
                mem_mhz: None
            })
            .unwrap(),
        o1
    );
    assert_eq!(replay.read().unwrap(), s1);
    assert_eq!(replay.throttle_reasons().unwrap(), t1);
    replay.unlock().unwrap();
    assert_eq!(replay.remaining(), 0);
}
