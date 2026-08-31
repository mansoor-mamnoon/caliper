//! Turning raw batch timings into a [`Record`].
//!
//! This is the pure heart of `bench()`: given the timing vectors a launcher
//! produced, per-batch throttle flags, the clock state, and the machine
//! fingerprint, [`reduce`] applies throttle invalidation and steady-state
//! trimming, computes the distribution, converts to per-launch numbers, and
//! assembles the record with the right advisory flags. No GPU, no clock, no I/O.

use serde::{Deserialize, Serialize};

use crate::schema::{Clocks, KernelLabel, Machine, Record};
use crate::stats::{summarize, Summary};
use crate::warmup::WarmupPlan;

/// The 2 MiB granularity the flush buffer is aligned to.
pub const FLUSH_GRANULARITY: u64 = 2 * 1024 * 1024;

/// Size, in bytes, of the buffer to stream through to evict the L2 cache
/// between measurements.
///
/// It is the device's L2 size rounded up to [`FLUSH_GRANULARITY`] plus one more
/// granule of headroom, so the whole cache is displaced even with allocation
/// slop. This is deliberately *not* a fixed constant: a hardcoded 256 MiB (a
/// common choice) is ~4x too large on a 72 MiB L2 and wastes time, while being
/// too small silently under-flushes.
#[must_use]
pub fn flush_buffer_bytes(l2_bytes: u64) -> u64 {
    let l2 = l2_bytes.max(FLUSH_GRANULARITY);
    let rounded = l2.div_ceil(FLUSH_GRANULARITY) * FLUSH_GRANULARITY;
    rounded + FLUSH_GRANULARITY
}

/// Result of [`invalidate`].
#[derive(Debug, Clone, PartialEq)]
pub struct Kept {
    /// GPU-event times for the batches that survived.
    pub gpu_us: Vec<f64>,
    /// Wall times for the same batches.
    pub wall_us: Vec<f64>,
    /// How many batches were dropped.
    pub n_invalidated: usize,
}

/// Drop batches whose `throttled` flag is set. An empty `throttled` slice means
/// nothing was throttled and everything is kept.
///
/// # Errors
/// Returns [`PipelineError::LengthMismatch`] if the slices disagree in length.
pub fn invalidate(
    gpu_us: &[f64],
    wall_us: &[f64],
    throttled: &[bool],
) -> Result<Kept, PipelineError> {
    if gpu_us.len() != wall_us.len() {
        return Err(PipelineError::LengthMismatch);
    }
    if !throttled.is_empty() && throttled.len() != gpu_us.len() {
        return Err(PipelineError::LengthMismatch);
    }

    if throttled.is_empty() {
        return Ok(Kept {
            gpu_us: gpu_us.to_vec(),
            wall_us: wall_us.to_vec(),
            n_invalidated: 0,
        });
    }

    let mut kept_gpu = Vec::new();
    let mut kept_wall = Vec::new();
    let mut dropped = 0;
    for i in 0..gpu_us.len() {
        if throttled[i] {
            dropped += 1;
        } else {
            kept_gpu.push(gpu_us[i]);
            kept_wall.push(wall_us[i]);
        }
    }
    Ok(Kept {
        gpu_us: kept_gpu,
        wall_us: kept_wall,
        n_invalidated: dropped,
    })
}

/// Everything [`reduce`] needs. Timings are per *batch* (of `batch` launches);
/// [`reduce`] converts them to per-launch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReduceInput {
    /// CUDA-event time per batch (microseconds).
    pub gpu_us: Vec<f64>,
    /// Wall time per batch (microseconds).
    pub wall_us: Vec<f64>,
    /// Launches per batch.
    pub batch: u32,
    /// Per-batch throttle flag; empty means none throttled.
    #[serde(default)]
    pub throttled: Vec<bool>,
    /// Throttle reasons observed while timing (NVML names).
    #[serde(default)]
    pub throttle_reasons: Vec<String>,
    /// How to handle warm-up: auto steady-state detection or a fixed trim.
    pub warmup: WarmupPlan,
    /// Whether the L2 flush was performed between samples.
    pub flush_l2: bool,
    /// Whether the clocks were locked for the run.
    pub clocks_locked: bool,
    /// Clock state to record.
    pub clocks: Clocks,
    /// Machine fingerprint to record.
    pub machine: Machine,
    /// Kernel identity to record.
    pub kernel: KernelLabel,
}

/// What can go wrong assembling a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineError {
    /// `batch` was zero.
    InvalidBatch,
    /// The timing / throttle slices disagreed in length.
    LengthMismatch,
    /// No samples at all.
    NoSamples,
    /// Every sample was invalidated by throttling.
    AllInvalidated,
    /// A sample was not finite.
    NonFinite,
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::InvalidBatch => "batch size must be non-zero",
            Self::LengthMismatch => "timing and throttle slices disagree in length",
            Self::NoSamples => "no timing samples",
            Self::AllInvalidated => "every sample was dropped for throttling",
            Self::NonFinite => "a timing sample was not finite",
        };
        f.write_str(s)
    }
}

impl std::error::Error for PipelineError {}

/// Assemble a [`Record`] from raw batch timings.
///
/// Steps: invalidate throttled batches, trim to steady state, convert
/// per-batch to per-launch, summarise, and set advisory flags
/// (`clocks-unlocked`, `throttled-samples-dropped`, `l2-flush-disabled`,
/// `warmup-not-converged`).
///
/// # Errors
/// See [`PipelineError`].
pub fn reduce(input: ReduceInput) -> Result<Record, PipelineError> {
    if input.batch == 0 {
        return Err(PipelineError::InvalidBatch);
    }
    if input.gpu_us.is_empty() {
        return Err(PipelineError::NoSamples);
    }
    if input
        .gpu_us
        .iter()
        .chain(&input.wall_us)
        .any(|x| !x.is_finite())
    {
        return Err(PipelineError::NonFinite);
    }

    let kept = invalidate(&input.gpu_us, &input.wall_us, &input.throttled)?;
    if kept.gpu_us.is_empty() {
        return Err(PipelineError::AllInvalidated);
    }

    let warm = input.warmup.resolve(&kept.gpu_us);
    let warm_gpu = &kept.gpu_us[warm.start..];
    let warm_wall = &kept.wall_us[warm.start..];
    if warm_gpu.is_empty() {
        return Err(PipelineError::NoSamples);
    }

    let b = f64::from(input.batch);
    let per_launch_gpu: Vec<f64> = warm_gpu.iter().map(|x| x / b).collect();
    let per_launch_wall: Vec<f64> = warm_wall.iter().map(|x| x / b).collect();

    let g: Summary = summarize(&per_launch_gpu).ok_or(PipelineError::NonFinite)?;
    let w: Summary = summarize(&per_launch_wall).ok_or(PipelineError::NonFinite)?;

    let mut record = Record {
        clocks: input.clocks,
        machine: input.machine,
        kernel: input.kernel,
        ..Record::default()
    };

    record.timing.p10_us = Some(g.p10);
    record.timing.p50_us = Some(g.p50);
    record.timing.p90_us = Some(g.p90);
    record.timing.mad_us = Some(g.mad);
    record.timing.wall_p50_us = Some(w.p50);
    record.timing.launch_overhead_us = Some((w.p50 - g.p50).max(0.0));
    record.timing.n_samples = Some(warm_gpu.len() as u64);
    record.timing.n_warmup_to_steady = Some(warm.start as u64);
    record.timing.invalidated_samples = Some(kept.n_invalidated as u64);

    let mut reasons = input.throttle_reasons.clone();
    reasons.sort();
    reasons.dedup();
    record.throttle_reasons = reasons;

    if !input.clocks_locked {
        record.flags.push("clocks-unlocked".to_string());
    }
    if kept.n_invalidated > 0 {
        record.flags.push("throttled-samples-dropped".to_string());
    }
    if !input.flush_l2 {
        record.flags.push("l2-flush-disabled".to_string());
    }
    if !warm.converged {
        record.flags.push("warmup-not-converged".to_string());
    }

    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    fn base_input(gpu_us: Vec<f64>, wall_us: Vec<f64>) -> ReduceInput {
        ReduceInput {
            gpu_us,
            wall_us,
            batch: 32,
            throttled: Vec::new(),
            throttle_reasons: Vec::new(),
            warmup: WarmupPlan::default(),
            flush_l2: true,
            clocks_locked: true,
            clocks: Clocks::default(),
            machine: Machine::default(),
            kernel: KernelLabel::default(),
        }
    }

    #[test]
    fn flush_buffer_is_near_l2_not_a_fixed_constant() {
        let l2 = 75_497_472; // 72 MiB, an RTX 4090
        let buf = flush_buffer_bytes(l2);
        assert!(buf > l2 && buf <= l2 + 4 * MIB, "{buf}");
        assert_ne!(buf, 256 * MIB);
        assert_eq!(buf % FLUSH_GRANULARITY, 0);
        // tiny / zero L2 still gets a usable buffer
        assert!(flush_buffer_bytes(0) >= FLUSH_GRANULARITY);
        // a Hopper-class 50 MiB L2 gets ~50, not 256
        assert!(flush_buffer_bytes(50 * MIB) < 60 * MIB);
    }

    #[test]
    fn invalidate_keeps_everything_when_no_throttle_info() {
        let k = invalidate(&[1.0, 2.0, 3.0], &[1.1, 2.1, 3.1], &[]).unwrap();
        assert_eq!(k.n_invalidated, 0);
        assert_eq!(k.gpu_us, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn invalidate_drops_flagged_batches() {
        let k = invalidate(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[false, true, false, true],
        )
        .unwrap();
        assert_eq!(k.n_invalidated, 2);
        assert_eq!(k.gpu_us, vec![1.0, 3.0]);
    }

    #[test]
    fn invalidate_rejects_length_mismatch() {
        assert_eq!(
            invalidate(&[1.0, 2.0], &[1.0], &[]),
            Err(PipelineError::LengthMismatch)
        );
        assert_eq!(
            invalidate(&[1.0, 2.0], &[1.0, 2.0], &[true]),
            Err(PipelineError::LengthMismatch)
        );
    }

    #[test]
    fn reduce_of_a_clean_locked_run_has_per_launch_timing_and_no_flags() {
        // 50 batches of 32 launches, ~6400us/batch -> 200us/launch.
        let gpu: Vec<f64> = (0..50).map(|i| 6400.0 + (i % 5) as f64).collect();
        let wall: Vec<f64> = gpu.iter().map(|g| g + 320.0).collect(); // 10us/launch overhead
        let rec = reduce(base_input(gpu, wall)).unwrap();

        let p50 = rec.timing.p50_us.unwrap();
        assert!((199.0..201.0).contains(&p50), "{p50}");
        assert!((9.0..11.0).contains(&rec.timing.launch_overhead_us.unwrap()));
        assert_eq!(rec.timing.invalidated_samples, Some(0));
        assert_eq!(rec.timing.n_warmup_to_steady, Some(0));
        assert!(rec.flags.is_empty(), "{:?}", rec.flags);
        assert!(rec.timing.n_samples.unwrap() >= 40);
    }

    #[test]
    fn reduce_flags_an_unlocked_run_with_dropped_throttled_batches() {
        let mut gpu: Vec<f64> = vec![6400.0; 40];
        gpu[10] = 9999.0;
        gpu[11] = 9999.0;
        let wall: Vec<f64> = gpu.iter().map(|g| g + 320.0).collect();
        let mut throttled = vec![false; 40];
        throttled[10] = true;
        throttled[11] = true;

        let mut input = base_input(gpu, wall);
        input.clocks_locked = false;
        input.flush_l2 = false;
        input.throttled = throttled;
        input.throttle_reasons = vec!["SW_POWER_CAP".into(), "SW_POWER_CAP".into()];

        let rec = reduce(input).unwrap();
        assert_eq!(rec.timing.invalidated_samples, Some(2));
        assert_eq!(rec.throttle_reasons, vec!["SW_POWER_CAP".to_string()]);
        let f = &rec.flags;
        assert!(f.contains(&"clocks-unlocked".to_string()));
        assert!(f.contains(&"throttled-samples-dropped".to_string()));
        assert!(f.contains(&"l2-flush-disabled".to_string()));
        // the 9999 outliers were dropped, so p50 is ~200/launch
        assert!((199.0..201.0).contains(&rec.timing.p50_us.unwrap()));
    }

    #[test]
    fn reduce_honours_a_fixed_warmup() {
        let gpu: Vec<f64> = vec![6400.0; 40];
        let wall: Vec<f64> = gpu.iter().map(|g| g + 320.0).collect();
        let mut input = base_input(gpu, wall);
        input.warmup = WarmupPlan::fixed(25);

        let rec = reduce(input).unwrap();
        assert_eq!(rec.timing.n_warmup_to_steady, Some(25));
        assert_eq!(rec.timing.n_samples, Some(15));
        assert!((199.0..201.0).contains(&rec.timing.p50_us.unwrap()));
    }

    #[test]
    fn reduce_trims_a_cold_ramp() {
        // 30 hot batches decaying, then 60 flat at 6400.
        let mut gpu: Vec<f64> = (0..30)
            .map(|i| 6400.0 + 4000.0 * (-(i as f64) / 8.0).exp())
            .collect();
        gpu.extend(std::iter::repeat_n(6400.0, 60));
        let wall: Vec<f64> = gpu.iter().map(|g| g + 320.0).collect();

        let rec = reduce(base_input(gpu, wall)).unwrap();
        assert!(rec.timing.n_warmup_to_steady.unwrap() > 0);
        assert!((199.0..201.0).contains(&rec.timing.p50_us.unwrap()));
        assert!(!rec.flags.contains(&"warmup-not-converged".to_string()));
    }

    #[test]
    fn reduce_rejects_bad_input() {
        assert_eq!(
            reduce(ReduceInput {
                batch: 0,
                ..base_input(vec![1.0], vec![1.0])
            }),
            Err(PipelineError::InvalidBatch)
        );
        assert_eq!(
            reduce(base_input(vec![], vec![])),
            Err(PipelineError::NoSamples)
        );
        assert_eq!(
            reduce(base_input(vec![1.0, f64::NAN], vec![1.0, 1.0])),
            Err(PipelineError::NonFinite)
        );
        let mut all_bad = base_input(vec![1.0, 2.0], vec![1.0, 2.0]);
        all_bad.throttled = vec![true, true];
        assert_eq!(reduce(all_bad), Err(PipelineError::AllInvalidated));
    }
}
