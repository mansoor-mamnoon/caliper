//! The device-layer ports.
//!
//! All methods take `&mut self`: the real implementations hold driver handles,
//! and the fixture player advances a cursor.

use caliper_core::schema::Machine;

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
