//! Errors from the device layer.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Anything that can go wrong talking to (or replaying) a device.
#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
pub enum GpuError {
    /// No CUDA device is available.
    #[error("no CUDA device available")]
    NoDevice,

    /// The driver refused the operation (missing privilege).
    #[error("operation not permitted by the driver: {0}")]
    PermissionDenied(String),

    /// The device or driver does not support the operation.
    #[error("not supported on this device/driver: {0}")]
    Unsupported(String),

    /// An NVML call failed.
    #[error("NVML error: {0}")]
    Nvml(String),

    /// A CUDA call failed.
    #[error("CUDA error: {0}")]
    Cuda(String),

    /// A recording ran out of entries mid-session.
    #[error("fixture exhausted: expected {port}::{method}, recording has no more entries")]
    FixtureExhausted {
        /// Port the caller expected to call.
        port: String,
        /// Method the caller expected to call.
        method: String,
    },

    /// A recorded call did not match what the caller actually did.
    #[error("fixture mismatch: expected {expected}, got {actual}")]
    FixtureMismatch {
        /// What the recording says should happen next.
        expected: String,
        /// What the caller did.
        actual: String,
    },

    /// Reading or parsing a recording failed.
    #[error("fixture I/O error: {0}")]
    FixtureIo(String),
}

/// Convenience alias for device-layer results.
pub type Result<T> = std::result::Result<T, GpuError>;
