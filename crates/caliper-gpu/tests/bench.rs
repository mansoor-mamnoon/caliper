//! L1: `bench()` end to end over a recorded device session, no GPU.

use std::path::PathBuf;

use caliper_gpu::{run_replay, BenchOpts, GpuError};

fn fixture(name: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/bench")
        .join(name);
    std::fs::read_to_string(p).expect("fixture reads")
}

fn opts(batches: u32) -> BenchOpts {
    BenchOpts {
        batches,
        ..BenchOpts::default()
    }
}

#[test]
fn happy_run_produces_a_populated_locked_record() {
    let rec = run_replay(&fixture("happy.jsonl"), &opts(40)).unwrap();

    let t = &rec.timing;
    assert!(
        (198.0..202.0).contains(&t.p50_us.unwrap()),
        "{:?}",
        t.p50_us
    );
    assert!(t.p10_us.unwrap() <= t.p50_us.unwrap());
    assert!(t.p50_us.unwrap() <= t.p90_us.unwrap());
    assert!((8.0..12.0).contains(&t.launch_overhead_us.unwrap()));
    assert_eq!(t.invalidated_samples, Some(0));
    assert_eq!(t.n_warmup_to_steady, Some(0));
    assert_eq!(t.n_samples, Some(40));

    assert!(rec.flags.is_empty(), "unexpected flags: {:?}", rec.flags);
    assert_eq!(rec.clocks.locked, Some(true));
    assert_eq!(rec.clocks.sm_mhz, Some(2520));
    assert_eq!(rec.machine.sm_arch.as_deref(), Some("sm_89"));
    assert_eq!(rec.machine.l2_bytes, Some(75_497_472));
    assert_eq!(rec.kernel.name.as_deref(), Some("kernel"));
    assert_eq!(rec.schema_version, "1");
    // ptxas came from the module probe
    assert_eq!(rec.ptxas.regs_per_thread, Some(168));
    assert_eq!(rec.ptxas.smem_static_bytes, Some(99328));
}

#[test]
fn a_run_with_no_ptxas_is_flagged_but_not_failed() {
    let rec = run_replay(&fixture("ptxas_unavailable.jsonl"), &opts(40)).unwrap();
    assert!(
        rec.flags.contains(&"ptxas-unavailable".to_string()),
        "{:?}",
        rec.flags
    );
    assert_eq!(rec.ptxas.regs_per_thread, None);
    assert!((198.0..202.0).contains(&rec.timing.p50_us.unwrap())); // timing still fine
}

#[test]
fn a_multi_kernel_probe_picks_the_one_matching_the_key() {
    let rec = run_replay(&fixture("multi_kernel_probe.jsonl"), &opts(40)).unwrap();
    // the module also has "epilogue_helper" (32 regs); the "kernel"-named one wins
    assert_eq!(rec.ptxas.regs_per_thread, Some(200));
    assert_eq!(rec.ptxas.smem_static_bytes, Some(40960));
    assert!(!rec.flags.contains(&"ptxas-unavailable".to_string()));
}

#[test]
fn a_hard_probe_error_propagates() {
    let err = run_replay(&fixture("probe_hard_error.jsonl"), &opts(40)).unwrap_err();
    assert!(matches!(err, GpuError::Cuda(_)), "{err:?}");
}

#[test]
fn unlocked_run_with_throttling_is_flagged_and_cleaned() {
    let rec = run_replay(&fixture("unlocked_throttled.jsonl"), &opts(40)).unwrap();

    assert_eq!(rec.timing.invalidated_samples, Some(2));
    assert_eq!(rec.timing.n_samples, Some(38));
    // the two 20000 us outliers were dropped, so p50 is back at ~200/launch
    assert!((198.0..202.0).contains(&rec.timing.p50_us.unwrap()));
    assert_eq!(rec.throttle_reasons, vec!["SW_POWER_CAP".to_string()]);

    let f = &rec.flags;
    assert!(f.contains(&"clocks-unlocked".to_string()), "{f:?}");
    assert!(
        f.contains(&"throttled-samples-dropped".to_string()),
        "{f:?}"
    );
    assert_eq!(rec.clocks.locked, Some(false));
}

#[test]
fn cold_ramp_run_is_trimmed_to_steady_state() {
    let rec = run_replay(&fixture("cold_ramp.jsonl"), &opts(70)).unwrap();

    let start = rec.timing.n_warmup_to_steady.unwrap();
    assert!(
        start > 0,
        "expected the ramp to be trimmed, start = {start}"
    );
    assert!(rec.timing.n_samples.unwrap() < 70);
    assert!((198.0..203.0).contains(&rec.timing.p50_us.unwrap()));
    assert!(!rec.flags.contains(&"warmup-not-converged".to_string()));
}

#[test]
fn fewer_batches_than_recorded_is_an_arg_mismatch() {
    // The recording's time_batches args say batches:40; asking for 10 -> mismatch.
    let err = run_replay(&fixture("happy.jsonl"), &opts(10)).unwrap_err();
    assert!(matches!(err, GpuError::FixtureMismatch { .. }), "{err:?}");
}

#[test]
fn a_recording_with_a_leftover_call_is_rejected() {
    let err = run_replay(&fixture("trailing_call.jsonl"), &opts(40)).unwrap_err();
    match err {
        GpuError::FixtureMismatch { expected, .. } => {
            assert_eq!(expected, "recording fully consumed");
        }
        other => panic!("expected a 'fully consumed' mismatch, got {other:?}"),
    }
}

#[test]
fn a_hard_lock_error_degrades_to_an_unlocked_run() {
    let rec = run_replay(&fixture("lock_error.jsonl"), &opts(40)).unwrap();
    assert!(
        rec.flags.contains(&"clocks-unlocked".to_string()),
        "{:?}",
        rec.flags
    );
    assert_eq!(rec.clocks.locked, Some(false));
    assert!((198.0..202.0).contains(&rec.timing.p50_us.unwrap()));
}

#[test]
fn auto_graph_run_is_tagged_and_a_roofline_spec_is_recorded() {
    use caliper_core::RooflineSpec;

    let opts = BenchOpts {
        batches: 40,
        roofline: Some(RooflineSpec {
            dtype: "bf16".to_string(),
            flops: 2.0 * 4096.0 * 4096.0 * 4096.0,
            bytes_hbm: 3.0 * 4096.0 * 4096.0 * 2.0,
        }),
        ..BenchOpts::default()
    };
    let rec = run_replay(&fixture("graph_auto.jsonl"), &opts).unwrap();

    // launcher reported graph_used = true
    assert!(
        rec.flags.contains(&"graph-captured".to_string()),
        "{:?}",
        rec.flags
    );
    // sm_89 has no bf16 tensor peak? it does (165.2). roofline section is filled.
    assert!(rec.roofline.achieved_tflops.unwrap() > 0.0);
    assert!(rec.roofline.bound.is_some());
    assert!(rec.timing.mean_us.unwrap() > 0.0);
    assert!(rec.timing.min_us.unwrap() <= rec.timing.max_us.unwrap());
}

#[test]
fn a_fixed_warmup_trims_exactly_n_batches() {
    let rec = run_replay(
        &fixture("happy.jsonl"),
        &BenchOpts {
            batches: 40,
            warmup: caliper_core::WarmupPlan::fixed(12),
            ..BenchOpts::default()
        },
    )
    .unwrap();
    assert_eq!(rec.timing.n_warmup_to_steady, Some(12));
    assert_eq!(rec.timing.n_samples, Some(28));
}
