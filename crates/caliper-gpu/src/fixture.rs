//! Record and replay device-layer sessions.
//!
//! A *recording* is JSON Lines: one object per port call, in order.
//!
//! ```jsonl
//! {"port":"device_info","method":"snapshot","args":null,"ret":{"Ok":{ ... }}}
//! {"port":"gpu_clock","method":"lock","args":{"sm_mhz":2520,"mem_mhz":null},"ret":{"Ok":"Locked"}}
//! ```
//!
//! [`FixturePlayer`] implements every port by popping the next recorded call,
//! checking it matches what the caller did, and returning the recorded result.
//! [`Recorder`] wraps a real port and appends a line per call.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use caliper_core::schema::Machine;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{GpuError, Result};
use crate::ports::{DeviceInfo, GpuClock, KernelLauncher};
use crate::types::{ClockState, ClockTarget, LaunchSpec, LockOutcome, RawSamples};

/// One recorded port call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Call {
    /// Port name: `"kernel_launcher"`, `"gpu_clock"`, or `"device_info"`.
    pub port: String,
    /// Method name on that port.
    pub method: String,
    /// Serialised arguments (`null` when there are none, or when not checked).
    #[serde(default)]
    pub args: Value,
    /// Serialised `Result<ReturnType, GpuError>` — `{"Ok": ...}` / `{"Err": ...}`.
    pub ret: Value,
}

/// Replays a recording as any of the ports.
#[derive(Debug, Clone)]
pub struct FixturePlayer {
    calls: VecDeque<Call>,
    check_args: bool,
}

impl FixturePlayer {
    /// Parse a recording from JSON Lines text. Blank lines and lines starting
    /// with `#` (headers / comments, e.g. `# caliper-fixture v=0.0.1`) are
    /// ignored.
    pub fn from_jsonl(text: &str) -> Result<Self> {
        let mut calls = VecDeque::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let call: Call = serde_json::from_str(line)
                .map_err(|e| GpuError::FixtureIo(format!("line {}: {e}", i + 1)))?;
            calls.push_back(call);
        }
        Ok(Self {
            calls,
            check_args: true,
        })
    }

    /// Load a recording from a file.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|e| GpuError::FixtureIo(format!("{}: {e}", path.as_ref().display())))?;
        Self::from_jsonl(&text)
    }

    /// Stop checking that recorded arguments match the caller's arguments.
    #[must_use]
    pub fn without_arg_checks(mut self) -> Self {
        self.check_args = false;
        self
    }

    /// Recorded calls not yet consumed.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.calls.len()
    }

    fn next_ret<T: DeserializeOwned>(
        &mut self,
        port: &str,
        method: &str,
        args: Value,
    ) -> Result<T> {
        let call = self
            .calls
            .pop_front()
            .ok_or_else(|| GpuError::FixtureExhausted {
                port: port.to_string(),
                method: method.to_string(),
            })?;

        if call.port != port || call.method != method {
            return Err(GpuError::FixtureMismatch {
                expected: format!("{}::{}", call.port, call.method),
                actual: format!("{port}::{method}"),
            });
        }
        if self.check_args && !call.args.is_null() && call.args != args {
            return Err(GpuError::FixtureMismatch {
                expected: format!("{}::{} args {}", port, method, call.args),
                actual: format!("{port}::{method} args {args}"),
            });
        }

        let decoded: std::result::Result<T, GpuError> = serde_json::from_value(call.ret)
            .map_err(|e| GpuError::FixtureIo(format!("decoding {port}::{method} return: {e}")))?;
        decoded
    }
}

impl KernelLauncher for FixturePlayer {
    fn time_batches(&mut self, spec: &LaunchSpec) -> Result<RawSamples> {
        let args = serde_json::to_value(spec).unwrap_or(Value::Null);
        self.next_ret("kernel_launcher", "time_batches", args)
    }
}

impl GpuClock for FixturePlayer {
    fn lock(&mut self, target: ClockTarget) -> Result<LockOutcome> {
        let args = serde_json::to_value(target).unwrap_or(Value::Null);
        self.next_ret("gpu_clock", "lock", args)
    }
    fn unlock(&mut self) -> Result<()> {
        self.next_ret("gpu_clock", "unlock", Value::Null)
    }
    fn read(&mut self) -> Result<ClockState> {
        self.next_ret("gpu_clock", "read", Value::Null)
    }
    fn throttle_reasons(&mut self) -> Result<Vec<String>> {
        self.next_ret("gpu_clock", "throttle_reasons", Value::Null)
    }
}

impl DeviceInfo for FixturePlayer {
    fn snapshot(&mut self) -> Result<Machine> {
        self.next_ret("device_info", "snapshot", Value::Null)
    }
}

/// Wraps a real port and appends a recording line per call.
pub struct Recorder<T> {
    inner: T,
    sink: File,
}

impl<T> Recorder<T> {
    /// Wrap `inner`, appending recorded calls to `path` (created if absent).
    pub fn new(inner: T, path: impl AsRef<Path>) -> Result<Self> {
        let sink = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
            .map_err(|e| GpuError::FixtureIo(format!("{}: {e}", path.as_ref().display())))?;
        Ok(Self { inner, sink })
    }

    /// Unwrap the recorder, returning the inner port.
    pub fn into_inner(self) -> T {
        self.inner
    }

    fn write_call<A, R>(
        &mut self,
        port: &str,
        method: &str,
        args: &A,
        ret: &std::result::Result<R, GpuError>,
    ) -> Result<()>
    where
        A: Serialize,
        R: Serialize,
    {
        let call = Call {
            port: port.to_string(),
            method: method.to_string(),
            args: serde_json::to_value(args).unwrap_or(Value::Null),
            ret: serde_json::to_value(ret)
                .map_err(|e| GpuError::FixtureIo(format!("encoding {port}::{method}: {e}")))?,
        };
        let line = serde_json::to_string(&call).map_err(|e| GpuError::FixtureIo(e.to_string()))?;
        writeln!(self.sink, "{line}").map_err(|e| GpuError::FixtureIo(e.to_string()))
    }
}

impl<T: KernelLauncher> KernelLauncher for Recorder<T> {
    fn time_batches(&mut self, spec: &LaunchSpec) -> Result<RawSamples> {
        let ret = self.inner.time_batches(spec);
        self.write_call("kernel_launcher", "time_batches", spec, &ret)?;
        ret
    }
}

impl<T: GpuClock> GpuClock for Recorder<T> {
    fn lock(&mut self, target: ClockTarget) -> Result<LockOutcome> {
        let ret = self.inner.lock(target);
        self.write_call("gpu_clock", "lock", &target, &ret)?;
        ret
    }
    fn unlock(&mut self) -> Result<()> {
        let ret = self.inner.unlock();
        self.write_call("gpu_clock", "unlock", &(), &ret)?;
        ret
    }
    fn read(&mut self) -> Result<ClockState> {
        let ret = self.inner.read();
        self.write_call("gpu_clock", "read", &(), &ret)?;
        ret
    }
    fn throttle_reasons(&mut self) -> Result<Vec<String>> {
        let ret = self.inner.throttle_reasons();
        self.write_call("gpu_clock", "throttle_reasons", &(), &ret)?;
        ret
    }
}

impl<T: DeviceInfo> DeviceInfo for Recorder<T> {
    fn snapshot(&mut self) -> Result<Machine> {
        let ret = self.inner.snapshot();
        self.write_call("device_info", "snapshot", &(), &ret)?;
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_and_comment_lines_are_ignored() {
        let jsonl = concat!(
            "# caliper-fixture v=0.0.1 arch=sm_89\n",
            "\n  \n",
            "{\"port\":\"gpu_clock\",\"method\":\"unlock\",\"ret\":{\"Ok\":null}}\n",
            "\n",
        );
        let p = FixturePlayer::from_jsonl(jsonl).unwrap();
        assert_eq!(p.remaining(), 1);
    }

    #[test]
    fn malformed_line_is_a_fixture_io_error() {
        let err = FixturePlayer::from_jsonl("{not json").unwrap_err();
        assert!(matches!(err, GpuError::FixtureIo(_)));
    }

    #[test]
    fn exhausted_recording_reports_the_expected_call() {
        let mut p = FixturePlayer::from_jsonl("").unwrap();
        let err = GpuClock::read(&mut p).unwrap_err();
        assert_eq!(
            err,
            GpuError::FixtureExhausted {
                port: "gpu_clock".into(),
                method: "read".into()
            }
        );
    }

    #[test]
    fn wrong_call_order_is_a_mismatch() {
        let jsonl = r#"{"port":"gpu_clock","method":"unlock","args":null,"ret":{"Ok":null}}"#;
        let mut p = FixturePlayer::from_jsonl(jsonl).unwrap();
        let err = GpuClock::read(&mut p).unwrap_err();
        assert!(matches!(err, GpuError::FixtureMismatch { .. }));
    }
}
