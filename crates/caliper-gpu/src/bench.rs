//! `bench()` v0: drive the device layer in order and reduce the result.
//!
//! [`run`] takes one object that implements all three ports (a "device layer")
//! and calls them in a fixed sequence: snapshot the device, lock the clocks
//! (unless told not to), time the batches, read the final clock state, unlock,
//! then hand everything to [`caliper_core::reduce`]. Passing a
//! [`crate::fixture::FixturePlayer`] replays a recorded session with no GPU.

use caliper_core::schema::{KernelLabel, Record};
use caliper_core::warmup::WarmupOpts;
use caliper_core::{reduce, ReduceInput};
use serde::{Deserialize, Serialize};

use crate::error::{GpuError, Result};
use crate::fixture::FixturePlayer;
use crate::ports::{DeviceInfo, GpuClock, KernelLauncher};
use crate::types::{ClockTarget, LaunchSpec, LockOutcome};

/// A device layer is anything that can launch kernels, control clocks, and
/// describe the device. The real backend and the fixture player both satisfy it.
pub trait DeviceLayer: KernelLauncher + GpuClock + DeviceInfo {}
impl<T: KernelLauncher + GpuClock + DeviceInfo> DeviceLayer for T {}

/// Options for one `bench()` run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BenchOpts {
    /// Opaque kernel key, forwarded in the [`LaunchSpec`] and recorded on the
    /// kernel label.
    pub kernel_key: String,
    /// Launches per timed batch.
    pub batch: u32,
    /// Number of batches to time.
    pub batches: u32,
    /// Capture each batch into a CUDA graph and replay it.
    pub cuda_graph: bool,
    /// Flush the L2 cache between batches.
    pub flush_l2: bool,
    /// Attempt to lock the clocks.
    pub lock_clocks: bool,
    /// Clock targets when locking.
    pub clock_target: ClockTarget,
    /// Steady-state detection options.
    pub warmup: WarmupOpts,
    /// Element dtype, recorded on the kernel label.
    pub dtype: Option<String>,
    /// Implementation family, recorded on the kernel label.
    pub kernel_impl: Option<String>,
}

impl Default for BenchOpts {
    fn default() -> Self {
        Self {
            kernel_key: "kernel".to_string(),
            batch: 32,
            batches: 50,
            cuda_graph: false,
            flush_l2: true,
            lock_clocks: true,
            clock_target: ClockTarget::default(),
            warmup: WarmupOpts::default(),
            dtype: None,
            kernel_impl: None,
        }
    }
}

/// Run one measurement through `layer` and return the assembled record.
///
/// # Errors
/// Propagates any port error, and any [`caliper_core::PipelineError`] from the
/// reduction (as [`GpuError::Unsupported`] carrying the message).
pub fn run<L: DeviceLayer>(layer: &mut L, opts: &BenchOpts) -> Result<Record> {
    let machine = layer.snapshot()?;

    let clocks_locked = if opts.lock_clocks {
        matches!(layer.lock(opts.clock_target)?, LockOutcome::Locked)
    } else {
        false
    };

    let spec = LaunchSpec {
        kernel_key: opts.kernel_key.clone(),
        batch: opts.batch,
        batches: opts.batches,
        use_graph: opts.cuda_graph,
    };
    let raw = layer.time_batches(&spec)?;

    let clock_state = layer.read()?;
    if clocks_locked {
        let _ = layer.unlock();
    }

    let input = ReduceInput {
        gpu_us: raw.gpu_us,
        wall_us: raw.wall_us,
        batch: raw.batch,
        throttled: raw.throttled,
        throttle_reasons: raw.throttle_reasons,
        warmup: opts.warmup,
        flush_l2: opts.flush_l2,
        clocks_locked,
        clocks: clock_state.into(),
        machine,
        kernel: KernelLabel {
            name: Some(opts.kernel_key.clone()),
            r#impl: opts.kernel_impl.clone(),
            dtype: opts.dtype.clone(),
            ..KernelLabel::default()
        },
    };

    reduce(input).map_err(|e| GpuError::Unsupported(format!("reduction failed: {e}")))
}

/// Drive a recorded session (JSON Lines) through [`run`].
///
/// The recording must hold, in order: `device_info::snapshot`,
/// `gpu_clock::lock` (only if `opts.lock_clocks`), `kernel_launcher::time_batches`,
/// `gpu_clock::read`, and `gpu_clock::unlock` (only if the lock succeeded).
///
/// # Errors
/// As [`run`], plus a fixture parse error.
pub fn run_replay(recording: &str, opts: &BenchOpts) -> Result<Record> {
    let mut layer = FixturePlayer::from_jsonl(recording)?;
    let record = run(&mut layer, opts)?;
    if layer.remaining() != 0 {
        return Err(GpuError::FixtureMismatch {
            expected: "recording fully consumed".to_string(),
            actual: format!("{} call(s) left over", layer.remaining()),
        });
    }
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_opts_round_trip_json() {
        let o = BenchOpts::default();
        let j = serde_json::to_string(&o).unwrap();
        assert_eq!(serde_json::from_str::<BenchOpts>(&j).unwrap(), o);
    }

    #[test]
    fn bench_opts_json_is_lenient_about_missing_keys() {
        let o: BenchOpts = serde_json::from_str(r#"{"batch": 16}"#).unwrap();
        assert_eq!(o.batch, 16);
        assert_eq!(o.batches, 50); // default
    }
}
