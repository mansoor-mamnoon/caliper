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
    /// Per-batch throttle flag: `throttled[i]` is true if the GPU was throttling
    /// while batch `i` was timed. Empty means the launcher observed no
    /// throttling.
    #[serde(default)]
    pub throttled: Vec<bool>,
    /// Union of throttle reasons (NVML names) observed while timing.
    #[serde(default)]
    pub throttle_reasons: Vec<String>,
    /// Threads per block of the timed launch, if the launcher knows it. Drives
    /// the occupancy section and the driver occupancy-API cross-check.
    #[serde(default)]
    pub block_size: Option<u32>,
    /// Total grid blocks of the timed launch, for the occupancy wave count.
    #[serde(default)]
    pub grid_blocks: Option<u32>,
    /// Dynamic shared memory per block requested at launch (bytes).
    #[serde(default)]
    pub dynamic_smem_bytes: Option<u32>,
    /// GPU-event time of a single un-batched launch, when the launcher timed
    /// one to resolve `cuda_graph="auto"` (microseconds).
    #[serde(default)]
    pub single_launch_us: Option<f64>,
    /// Whether the launcher actually captured the batch into a CUDA graph.
    /// `None` when the launcher did not report it.
    #[serde(default)]
    pub graph_used: Option<bool>,
}

/// Whether to capture each timed batch into a CUDA graph and replay it, which
/// removes per-launch overhead from the measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GraphMode {
    /// Decide per kernel: capture a graph when a single launch is short enough
    /// that launch overhead would dominate. (The decision is made on-device.)
    #[default]
    Auto,
    /// Always capture a graph.
    On,
    /// Never capture a graph.
    Off,
}

impl GraphMode {
    /// The lowercase token (`"auto"` / `"on"` / `"off"`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }
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
