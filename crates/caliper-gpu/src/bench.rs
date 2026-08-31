//! `bench()` v0: drive the device layer in order and reduce the result.
//!
//! [`run`] takes one object that implements all three ports (a "device layer")
//! and calls them in a fixed sequence:
//!
//! 1. `device_info::snapshot`
//! 2. `gpu_clock::lock` (only if `opts.lock_clocks`; a refusal is not fatal)
//! 3. `gpu_clock::throttle_reasons` — the "before" throttle poll
//! 4. `kernel_launcher::time_batches`
//! 5. `gpu_clock::throttle_reasons` — the "after" throttle poll (per-batch
//!    "during" polling is the launcher's job, since only it knows batch
//!    boundaries, and it reports `RawSamples::throttled`)
//! 6. `gpu_clock::read`
//! 7. `gpu_clock::unlock` (only if the lock succeeded)
//! 8. [`caliper_core::reduce`]
//!
//! Passing a [`crate::fixture::FixturePlayer`] replays a recorded session with
//! no GPU.

use caliper_core::schema::{KernelLabel, Ptxas, Record};
use caliper_core::warmup::WarmupPlan;
use caliper_core::{reduce, ParsedKernel, ReduceInput};
use serde::{Deserialize, Serialize};

use crate::error::{GpuError, Result};
use crate::fixture::FixturePlayer;
use crate::ports::{DeviceInfo, GpuClock, KernelLauncher, ModuleProbe};
use crate::types::{ClockTarget, GraphMode, LaunchSpec, LockOutcome};

/// A device layer is anything that can launch kernels, control clocks, describe
/// the device, and inspect a compiled module. The real backend and the fixture
/// player both satisfy it.
pub trait DeviceLayer: KernelLauncher + GpuClock + DeviceInfo + ModuleProbe {}
impl<T: KernelLauncher + GpuClock + DeviceInfo + ModuleProbe + ?Sized> DeviceLayer for T {}

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
    /// CUDA-graph capture policy.
    pub cuda_graph: GraphMode,
    /// Flush the L2 cache between batches.
    pub flush_l2: bool,
    /// Attempt to lock the clocks.
    pub lock_clocks: bool,
    /// Clock targets when locking.
    pub clock_target: ClockTarget,
    /// How to handle warm-up: auto steady-state detection or a fixed trim.
    pub warmup: WarmupPlan,
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
            cuda_graph: GraphMode::Auto,
            flush_l2: true,
            lock_clocks: true,
            clock_target: ClockTarget::default(),
            warmup: WarmupPlan::default(),
            dtype: None,
            kernel_impl: None,
        }
    }
}

/// Run one measurement through `layer` and return the assembled record.
///
/// A clock-lock refusal (`Denied` / `Unsupported`, or a `PermissionDenied` /
/// `Unsupported` error) is not fatal: the run proceeds unlocked and the record
/// is tagged `clocks-unlocked`. Genuine device errors still propagate.
///
/// # Errors
/// Propagates a device error, and any [`caliper_core::PipelineError`] from the
/// reduction (as [`GpuError::Unsupported`] carrying the message).
pub fn run<L: DeviceLayer + ?Sized>(layer: &mut L, opts: &BenchOpts) -> Result<Record> {
    let machine = layer.snapshot()?;

    let clocks_locked = if opts.lock_clocks {
        match layer.lock(opts.clock_target) {
            Ok(LockOutcome::Locked) => true,
            Ok(LockOutcome::Denied | LockOutcome::Unsupported) => false,
            Err(GpuError::PermissionDenied(_) | GpuError::Unsupported(_)) => false,
            Err(other) => return Err(other),
        }
    } else {
        false
    };

    // GraphMode::Auto is resolved on-device by timing a single launch; until the
    // real launcher exists it behaves as Off.
    let use_graph = matches!(opts.cuda_graph, GraphMode::On);

    let mut throttle_reasons = layer.throttle_reasons()?; // "before" poll

    let spec = LaunchSpec {
        kernel_key: opts.kernel_key.clone(),
        batch: opts.batch,
        batches: opts.batches,
        use_graph,
    };
    let raw = layer.time_batches(&spec)?;

    throttle_reasons.extend(layer.throttle_reasons()?); // "after" poll
    throttle_reasons.extend(raw.throttle_reasons);

    // Static resource usage. A probe that isn't available (no ptxas, no cuda
    // feature) is not fatal -- the record is flagged `ptxas-unavailable`.
    let ptxas = match layer.probe(&opts.kernel_key) {
        Ok(kernels) => pick_ptxas(&kernels, &opts.kernel_key),
        Err(GpuError::Unsupported(_) | GpuError::PermissionDenied(_) | GpuError::NoDevice) => None,
        Err(other) => return Err(other),
    };

    let clock_state = layer.read()?;
    if clocks_locked {
        let _ = layer.unlock();
    }

    let input = ReduceInput {
        gpu_us: raw.gpu_us,
        wall_us: raw.wall_us,
        batch: raw.batch,
        throttled: raw.throttled,
        throttle_reasons,
        warmup: opts.warmup,
        flush_l2: opts.flush_l2,
        clocks_locked,
        clocks: clock_state.into(),
        machine,
        ptxas,
        kernel: KernelLabel {
            name: Some(opts.kernel_key.clone()),
            r#impl: opts.kernel_impl.clone(),
            dtype: opts.dtype.clone(),
            ..KernelLabel::default()
        },
        // Launch geometry and a roofline spec are not yet reported by the
        // launcher port; the on-device path fills these in. Until then the
        // occupancy and roofline sections stay empty.
        block_size: None,
        grid_blocks: None,
        roofline: None,
    };

    reduce(input).map_err(|e| GpuError::Unsupported(format!("reduction failed: {e}")))
}

/// From a probed module, the resource usage of the kernel matching `kernel_key`,
/// or the first kernel, or `None` if the module had no kernels.
fn pick_ptxas(kernels: &[ParsedKernel], kernel_key: &str) -> Option<Ptxas> {
    if kernels.is_empty() {
        return None;
    }
    let chosen = kernels
        .iter()
        .find(|k| k.name.as_deref() == Some(kernel_key))
        .unwrap_or(&kernels[0]);
    Some(chosen.ptxas.clone())
}

/// Drive a recorded session (JSON Lines) through [`run`], then require the
/// recording to be fully consumed.
///
/// # Errors
/// As [`run`], plus a fixture parse error, or [`GpuError::FixtureMismatch`] if
/// the recording has calls left over.
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
        assert_eq!(o.cuda_graph, GraphMode::Auto); // default
    }

    #[test]
    fn graph_mode_and_warmup_serialise_as_expected() {
        let o = BenchOpts {
            cuda_graph: GraphMode::Off,
            warmup: WarmupPlan::fixed(25),
            ..BenchOpts::default()
        };
        let j = serde_json::to_value(&o).unwrap();
        assert_eq!(j["cuda_graph"], "off");
        assert_eq!(j["warmup"]["fixed"], 25);
        assert_eq!(serde_json::from_value::<BenchOpts>(j).unwrap(), o);
    }
}
