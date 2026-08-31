//! On-device implementations of the ports.
//!
//! Compiled only with `--features cuda`. The bodies here are typed stubs so the
//! trait objects and constructors exist on every platform and CI can
//! compile-check them; the real `cudarc` / `nvml-wrapper` calls are filled in and
//! validated on a CUDA host (see `docs/plan.md`, the Week 1 device tasks).
//!
//! Each stub returns [`GpuError::Unsupported`] rather than `todo!()` so a caller
//! on a machine without the implementation gets a clean, catchable failure.

use caliper_core::schema::Machine;

use crate::error::{GpuError, Result};
use crate::ports::{DeviceInfo, GpuClock, KernelLauncher};
use crate::types::{ClockState, ClockTarget, LaunchSpec, LockOutcome, RawSamples};

fn pending(what: &str) -> GpuError {
    GpuError::Unsupported(format!(
        "{what} is implemented and validated on a CUDA host"
    ))
}

/// CUDA-event kernel launcher and timer.
#[derive(Debug, Default)]
pub struct CudaLauncher {
    /// Ordinal of the device to launch on.
    pub device: u32,
}

impl CudaLauncher {
    /// Open the launcher for `device`.
    pub fn open(device: u32) -> Result<Self> {
        Ok(Self { device })
    }
}

impl KernelLauncher for CudaLauncher {
    fn time_batches(&mut self, _spec: &LaunchSpec) -> Result<RawSamples> {
        Err(pending("CUDA-event batch timing"))
    }
}

/// NVML-backed clock control and throttle reporting.
#[derive(Debug, Default)]
pub struct NvmlClock {
    /// Ordinal of the device.
    pub device: u32,
}

impl NvmlClock {
    /// Open NVML for `device`.
    pub fn open(device: u32) -> Result<Self> {
        Ok(Self { device })
    }
}

impl GpuClock for NvmlClock {
    fn lock(&mut self, _target: ClockTarget) -> Result<LockOutcome> {
        Err(pending("NVML clock locking"))
    }
    fn unlock(&mut self) -> Result<()> {
        Err(pending("NVML clock unlock"))
    }
    fn read(&mut self) -> Result<ClockState> {
        Err(pending("NVML clock read"))
    }
    fn throttle_reasons(&mut self) -> Result<Vec<String>> {
        Err(pending("NVML throttle-reason decode"))
    }
}

/// NVML + CUDA device-info snapshot.
#[derive(Debug, Default)]
pub struct NvmlDeviceInfo {
    /// Ordinal of the device.
    pub device: u32,
}

impl NvmlDeviceInfo {
    /// Open NVML for `device`.
    pub fn open(device: u32) -> Result<Self> {
        Ok(Self { device })
    }
}

impl DeviceInfo for NvmlDeviceInfo {
    fn snapshot(&mut self) -> Result<Machine> {
        Err(pending("NVML device-info snapshot"))
    }
}

/// The three real ports for one device, bundled so they satisfy
/// [`crate::bench::DeviceLayer`].
#[derive(Debug, Default)]
pub struct CudaDeviceLayer {
    launcher: CudaLauncher,
    clock: NvmlClock,
    info: NvmlDeviceInfo,
}

impl CudaDeviceLayer {
    /// Open every port for `device`.
    pub fn open(device: u32) -> Result<Self> {
        Ok(Self {
            launcher: CudaLauncher::open(device)?,
            clock: NvmlClock::open(device)?,
            info: NvmlDeviceInfo::open(device)?,
        })
    }
}

impl KernelLauncher for CudaDeviceLayer {
    fn time_batches(&mut self, spec: &LaunchSpec) -> Result<RawSamples> {
        self.launcher.time_batches(spec)
    }
}

impl GpuClock for CudaDeviceLayer {
    fn lock(&mut self, target: ClockTarget) -> Result<LockOutcome> {
        self.clock.lock(target)
    }
    fn unlock(&mut self) -> Result<()> {
        self.clock.unlock()
    }
    fn read(&mut self) -> Result<ClockState> {
        self.clock.read()
    }
    fn throttle_reasons(&mut self) -> Result<Vec<String>> {
        self.clock.throttle_reasons()
    }
}

impl DeviceInfo for CudaDeviceLayer {
    fn snapshot(&mut self) -> Result<Machine> {
        self.info.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stubs_construct_and_report_pending() {
        assert!(matches!(
            CudaLauncher::open(0).unwrap().time_batches(&LaunchSpec {
                kernel_key: "k".into(),
                batch: 1,
                batches: 1,
                use_graph: false,
            }),
            Err(GpuError::Unsupported(_))
        ));
        assert!(matches!(
            NvmlClock::open(0).unwrap().read(),
            Err(GpuError::Unsupported(_))
        ));
        assert!(matches!(
            NvmlDeviceInfo::open(0).unwrap().snapshot(),
            Err(GpuError::Unsupported(_))
        ));
    }
}
