//! Choosing a device layer at runtime.
//!
//! `CALIPER_GPU_PORTS` selects the backend:
//!
//! | value | behaviour |
//! |-------|-----------|
//! | `real` (default) | talk to the device (requires the `cuda` build feature) |
//! | `fixture` | replay the recording at `CALIPER_GPU_FIXTURE` |
//! | `record` | wrap the real device, writing a recording to `CALIPER_GPU_FIXTURE` |
//!
//! [`DeviceLayerHandle`] is a concrete enum that implements all three ports, so
//! callers stay monomorphic. On a build without the `cuda` feature, `real` /
//! `record` resolve to [`DeviceLayerHandle::Unavailable`], whose every method
//! returns [`GpuError::NoDevice`].

use std::path::PathBuf;

use caliper_core::schema::Machine;
use caliper_core::ParsedKernel;

use crate::error::{GpuError, Result};
use crate::fixture::FixturePlayer;
use crate::ports::{DeviceInfo, GpuClock, KernelLauncher, ModuleProbe};
use crate::types::{ClockState, ClockTarget, LaunchSpec, LockOutcome, RawSamples};

/// Which backend to build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortSelection {
    /// Talk to the real device.
    Real,
    /// Replay the recording at this path.
    Fixture(PathBuf),
    /// Wrap the real device, writing a recording to this path.
    Record(PathBuf),
}

impl PortSelection {
    /// Read `CALIPER_GPU_PORTS` and `CALIPER_GPU_FIXTURE` from the environment.
    ///
    /// # Errors
    /// [`GpuError::Unsupported`] for an unrecognised mode; [`GpuError::FixtureIo`]
    /// if `fixture` / `record` is requested without a path.
    pub fn from_env() -> Result<Self> {
        Self::parse(
            std::env::var("CALIPER_GPU_PORTS").ok().as_deref(),
            std::env::var("CALIPER_GPU_FIXTURE").ok().as_deref(),
        )
    }

    /// The pure core of [`from_env`](Self::from_env), for testing.
    ///
    /// # Errors
    /// As [`from_env`](Self::from_env).
    pub fn parse(mode: Option<&str>, path: Option<&str>) -> Result<Self> {
        match mode.unwrap_or("real").trim().to_ascii_lowercase().as_str() {
            "real" => Ok(Self::Real),
            "fixture" => Ok(Self::Fixture(require_path(path, "fixture")?)),
            "record" => Ok(Self::Record(require_path(path, "record")?)),
            other => Err(GpuError::Unsupported(format!(
                "CALIPER_GPU_PORTS={other:?}; expected real, fixture, or record"
            ))),
        }
    }
}

fn require_path(path: Option<&str>, mode: &str) -> Result<PathBuf> {
    path.filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| {
            GpuError::FixtureIo(format!(
                "CALIPER_GPU_PORTS={mode} needs CALIPER_GPU_FIXTURE=<path>"
            ))
        })
}

/// A device layer chosen at runtime. Implements every port.
#[non_exhaustive]
pub enum DeviceLayerHandle {
    /// Replaying a recording.
    Fixture(FixturePlayer),
    /// The real device.
    #[cfg(feature = "cuda")]
    Real(crate::real::CudaDeviceLayer),
    /// The real device, being recorded.
    #[cfg(feature = "cuda")]
    Record(crate::fixture::Recorder<crate::real::CudaDeviceLayer>),
    /// `real` / `record` was requested but this build lacks the `cuda` feature.
    Unavailable,
}

/// Build the handle for an explicit [`PortSelection`].
///
/// # Errors
/// A fixture parse error, or a device-open error for `real` / `record`.
pub fn open(selection: PortSelection) -> Result<DeviceLayerHandle> {
    match selection {
        PortSelection::Fixture(path) => {
            Ok(DeviceLayerHandle::Fixture(FixturePlayer::from_path(path)?))
        }
        #[cfg(feature = "cuda")]
        PortSelection::Real => Ok(DeviceLayerHandle::Real(crate::real::CudaDeviceLayer::open(
            0,
        )?)),
        #[cfg(feature = "cuda")]
        PortSelection::Record(path) => Ok(DeviceLayerHandle::Record(
            crate::fixture::Recorder::new(crate::real::CudaDeviceLayer::open(0)?, path)?,
        )),
        #[cfg(not(feature = "cuda"))]
        PortSelection::Real | PortSelection::Record(_) => Ok(DeviceLayerHandle::Unavailable),
    }
}

/// Build the handle from `CALIPER_GPU_PORTS` / `CALIPER_GPU_FIXTURE`.
///
/// # Errors
/// As [`PortSelection::from_env`] and [`open`].
pub fn open_from_env() -> Result<DeviceLayerHandle> {
    open(PortSelection::from_env()?)
}

macro_rules! dispatch {
    ($self:expr, $method:ident ( $($arg:expr),* )) => {
        match $self {
            DeviceLayerHandle::Fixture(p) => p.$method($($arg),*),
            #[cfg(feature = "cuda")]
            DeviceLayerHandle::Real(p) => p.$method($($arg),*),
            #[cfg(feature = "cuda")]
            DeviceLayerHandle::Record(p) => p.$method($($arg),*),
            DeviceLayerHandle::Unavailable => Err(GpuError::NoDevice),
        }
    };
}

impl KernelLauncher for DeviceLayerHandle {
    fn time_batches(&mut self, spec: &LaunchSpec) -> Result<RawSamples> {
        dispatch!(self, time_batches(spec))
    }
}

impl GpuClock for DeviceLayerHandle {
    fn lock(&mut self, target: ClockTarget) -> Result<LockOutcome> {
        dispatch!(self, lock(target))
    }
    fn unlock(&mut self) -> Result<()> {
        dispatch!(self, unlock())
    }
    fn read(&mut self) -> Result<ClockState> {
        dispatch!(self, read())
    }
    fn throttle_reasons(&mut self) -> Result<Vec<String>> {
        dispatch!(self, throttle_reasons())
    }
}

impl DeviceInfo for DeviceLayerHandle {
    fn snapshot(&mut self) -> Result<Machine> {
        dispatch!(self, snapshot())
    }
}

impl ModuleProbe for DeviceLayerHandle {
    fn probe(&mut self, kernel_key: &str) -> Result<Vec<ParsedKernel>> {
        dispatch!(self, probe(kernel_key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults_to_real() {
        assert_eq!(
            PortSelection::parse(None, None).unwrap(),
            PortSelection::Real
        );
    }

    #[test]
    fn parse_is_case_insensitive_and_trims() {
        assert_eq!(
            PortSelection::parse(Some("  RECORD "), Some("/tmp/r.jsonl")).unwrap(),
            PortSelection::Record(PathBuf::from("/tmp/r.jsonl"))
        );
    }

    #[test]
    fn parse_requires_a_path_for_fixture_and_record() {
        assert!(matches!(
            PortSelection::parse(Some("fixture"), None),
            Err(GpuError::FixtureIo(_))
        ));
        assert!(matches!(
            PortSelection::parse(Some("record"), Some("")),
            Err(GpuError::FixtureIo(_))
        ));
    }

    #[test]
    fn parse_rejects_an_unknown_mode() {
        assert!(matches!(
            PortSelection::parse(Some("bogus"), None),
            Err(GpuError::Unsupported(_))
        ));
    }

    #[test]
    fn open_fixture_selection_gives_a_working_layer() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/bench/happy.jsonl");
        let mut handle = open(PortSelection::Fixture(path)).unwrap();
        let rec = crate::bench::run(
            &mut handle,
            &crate::bench::BenchOpts {
                batches: 40,
                ..crate::bench::BenchOpts::default()
            },
        )
        .unwrap();
        assert!((198.0..202.0).contains(&rec.timing.p50_us.unwrap()));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn real_is_unavailable_without_the_cuda_feature() {
        let mut handle = open(PortSelection::Real).unwrap();
        assert!(matches!(handle.snapshot(), Err(GpuError::NoDevice)));
    }
}
