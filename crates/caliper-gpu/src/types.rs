//! Data the ports exchange with the rest of caliper.

use caliper_core::schema::Clocks;
use serde::{Deserialize, Serialize};

/// What to launch and how many times for one timing call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchSpec {
    /// Opaque key identifying the kernel + config. Used to select a recording
    /// and to key any per-config caches.
    pub kernel_key: String,
    /// Back-to-back launches per timed batch.
    pub batch: u32,
    /// Number of batches to time.
    pub batches: u32,
    /// Capture the batch into a CUDA graph and replay it (removes per-launch
    /// overhead from the measurement).
    pub use_graph: bool,
}

/// Raw per-batch timings from the launcher, in microseconds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawSamples {
    /// CUDA-event time for each batch.
    pub gpu_us: Vec<f64>,
    /// Host wall time for each batch.
    pub wall_us: Vec<f64>,
    /// Launches per batch, echoed back so callers can convert to per-launch.
    pub batch: u32,
}

/// A clock-lock request. `None` means "leave this clock alone / use the max".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ClockTarget {
    /// SM clock to lock to (MHz).
    pub sm_mhz: Option<u32>,
    /// Memory clock to lock to (MHz).
    pub mem_mhz: Option<u32>,
}

/// Outcome of a lock attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LockOutcome {
    /// Clocks are now locked.
    Locked,
    /// The driver refused (no permission). The caller should proceed unlocked
    /// and tag the run `clocks-unlocked`.
    Denied,
    /// The device / driver cannot lock clocks at all.
    Unsupported,
}

/// Observed clock state. Converts into [`caliper_core::schema::Clocks`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockState {
    /// Current SM clock (MHz).
    pub sm_mhz: Option<u32>,
    /// Current memory clock (MHz).
    pub mem_mhz: Option<u32>,
    /// Whether clocks are currently locked by caliper.
    pub locked: bool,
    /// How they were locked, e.g. `"nvml"`.
    pub lock_method: Option<String>,
}

impl From<ClockState> for Clocks {
    fn from(c: ClockState) -> Self {
        Clocks {
            sm_mhz: c.sm_mhz,
            mem_mhz: c.mem_mhz,
            locked: Some(c.locked),
            lock_method: c.lock_method,
        }
    }
}
