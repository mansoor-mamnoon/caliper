//! The device-layer ports.
//!
//! All methods take `&mut self`: the real implementations hold driver handles,
//! and the fixture player advances a cursor.

use caliper_core::schema::Machine;
use caliper_core::ParsedKernel;

use crate::error::Result;
use crate::types::{ClockState, ClockTarget, LaunchSpec, LockOutcome, RawSamples};

/// Launches a kernel and returns raw batch timings.
pub trait KernelLauncher {
    /// Time `spec.batches` batches of `spec.batch` back-to-back launches.
    fn time_batches(&mut self, spec: &LaunchSpec) -> Result<RawSamples>;
}

/// Reads and controls GPU clocks and reports throttle reasons.
pub trait GpuClock {
    /// Attempt to lock the clocks. A refusal comes back as
    /// [`LockOutcome::Denied`], not an error.
    fn lock(&mut self, target: ClockTarget) -> Result<LockOutcome>;
    /// Release any lock caliper applied.
    fn unlock(&mut self) -> Result<()>;
    /// Read the current clock state.
    fn read(&mut self) -> Result<ClockState>;
    /// Current throttle reasons as NVML names (e.g. `"SW_POWER_CAP"`).
    fn throttle_reasons(&mut self) -> Result<Vec<String>>;
}

/// Snapshots static device / host information into a [`Machine`].
pub trait DeviceInfo {
    /// Collect the fingerprint for the current device.
    fn snapshot(&mut self) -> Result<Machine>;
}

/// Inspects a compiled kernel's static resource usage (via `ptxas -v` /
/// `cuobjdump`), one entry per kernel in the module.
pub trait ModuleProbe {
    /// Resource usage for the module `kernel_key` was compiled into.
    fn probe(&mut self, kernel_key: &str) -> Result<Vec<ParsedKernel>>;

    /// The driver's own `cuOccupancyMaxActiveBlocksPerMultiprocessor` for this
    /// kernel at `block_size` threads and `dynamic_smem_bytes` of dynamic
    /// shared memory.
    ///
    /// `Ok(None)` means the probe cannot answer (no device / no driver handle),
    /// in which case [`caliper_core::occupancy`]'s model is the only source.
    /// The default returns `Ok(None)`; the real CUDA probe overrides it (and,
    /// until the driver call is wired up on a CUDA host, returns a "pending"
    /// error that [`crate::bench::run`] treats as "not available").
    fn max_active_blocks_per_sm(
        &mut self,
        kernel_key: &str,
        block_size: u32,
        dynamic_smem_bytes: u32,
    ) -> Result<Option<u32>> {
        let _ = (kernel_key, block_size, dynamic_smem_bytes);
        Ok(None)
    }
}
