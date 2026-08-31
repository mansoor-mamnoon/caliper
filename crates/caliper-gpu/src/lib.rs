//! caliper's device layer.
//!
//! Three ports abstract every GPU interaction:
//!
//! * [`KernelLauncher`] — launch a kernel and return raw batch timings.
//! * [`GpuClock`] — lock / read clocks and report throttle reasons.
//! * [`DeviceInfo`] — snapshot static device / host information.
//!
//! Each port has a real implementation ([`real`], compiled with `--features
//! cuda` on a CUDA host) and a [`fixture::FixturePlayer`] that replays a recorded
//! session so the control flow can be exercised with no GPU.
//! [`fixture::Recorder`] wraps a real port and writes that recording.
//!
//! `caliper-core` depends on none of this — it never touches hardware. Only the
//! orchestration above the ports does.

pub mod bench;
pub mod corpus;
pub mod doctor;
pub mod error;
pub mod fixture;
pub mod ports;
pub mod select;
pub mod types;

#[cfg(feature = "cuda")]
pub mod real;

pub use bench::{run, run_replay, BenchOpts, DeviceLayer};
pub use error::{GpuError, Result};
pub use ports::{DeviceInfo, GpuClock, KernelLauncher, ModuleProbe};
pub use select::{open_from_env, DeviceLayerHandle, PortSelection};
pub use types::{ClockState, ClockTarget, GraphMode, LaunchSpec, LockOutcome, RawSamples};
