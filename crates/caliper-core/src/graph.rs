//! CUDA-graph capture decision for `cuda_graph="auto"`.
//!
//! Capturing a batch into a CUDA graph and replaying it removes the per-launch
//! CPU cost from the measurement. That only matters when a single launch is
//! short enough that the launch overhead is a meaningful fraction of it; for a
//! long kernel the overhead is in the noise and capture just adds complexity.
//!
//! The threshold is on the *single-launch* GPU time. The on-device launcher
//! times one launch first, then this decides; the pure function lives here so
//! the policy is one testable place.

/// Default single-launch threshold (microseconds). At or above this a launch is
/// long enough that per-launch overhead is negligible, so `auto` stays eager.
///
/// 50 us: a ~7 us launch overhead is ~14% of a 50 us kernel and ~1.4% of a
/// 500 us one -- the knee where graph capture stops earning its keep.
pub const DEFAULT_SINGLE_LAUNCH_THRESHOLD_US: f64 = 50.0;

/// What `cuda_graph="auto"` resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphChoice {
    /// Capture the batch into a graph and replay it.
    Capture,
    /// Launch eagerly, one kernel launch at a time.
    Eager,
    /// No single-launch estimate available yet -- caller keeps the default
    /// (eager) and may decide later.
    Unknown,
}

impl GraphChoice {
    /// The record flag for this choice (`None` for [`GraphChoice::Unknown`]).
    #[must_use]
    pub fn flag(self) -> Option<&'static str> {
        match self {
            Self::Capture => Some("graph-captured"),
            Self::Eager => Some("graph-eager"),
            Self::Unknown => None,
        }
    }
}

/// Whether a launch this short should be captured into a graph.
#[must_use]
pub fn should_capture(single_launch_us: f64, threshold_us: f64) -> bool {
    single_launch_us.is_finite() && single_launch_us > 0.0 && single_launch_us < threshold_us
}

/// Resolve an explicit / automatic graph mode to a concrete choice.
///
/// `mode` is `"on"` / `"off"` / `"auto"` (anything else is treated as `"auto"`).
/// `single_launch_us` is the launcher's one-launch probe, if it ran.
#[must_use]
pub fn resolve(mode: &str, single_launch_us: Option<f64>) -> GraphChoice {
    match mode.trim().to_ascii_lowercase().as_str() {
        "on" => GraphChoice::Capture,
        "off" => GraphChoice::Eager,
        _ => match single_launch_us {
            Some(t) if should_capture(t, DEFAULT_SINGLE_LAUNCH_THRESHOLD_US) => {
                GraphChoice::Capture
            }
            Some(_) => GraphChoice::Eager,
            None => GraphChoice::Unknown,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_launches_are_captured_long_ones_are_not() {
        assert!(should_capture(5.0, DEFAULT_SINGLE_LAUNCH_THRESHOLD_US));
        assert!(should_capture(49.9, DEFAULT_SINGLE_LAUNCH_THRESHOLD_US));
        assert!(!should_capture(50.0, DEFAULT_SINGLE_LAUNCH_THRESHOLD_US));
        assert!(!should_capture(500.0, DEFAULT_SINGLE_LAUNCH_THRESHOLD_US));
        // degenerate inputs never capture
        assert!(!should_capture(0.0, 50.0));
        assert!(!should_capture(-1.0, 50.0));
        assert!(!should_capture(f64::NAN, 50.0));
    }

    #[test]
    fn explicit_modes_ignore_the_probe() {
        assert_eq!(resolve("on", None), GraphChoice::Capture);
        assert_eq!(resolve("on", Some(999.0)), GraphChoice::Capture);
        assert_eq!(resolve("off", Some(1.0)), GraphChoice::Eager);
    }

    #[test]
    fn auto_needs_the_probe_and_uses_the_threshold() {
        assert_eq!(resolve("auto", None), GraphChoice::Unknown);
        assert_eq!(resolve("AUTO", Some(4.0)), GraphChoice::Capture);
        assert_eq!(resolve("auto", Some(120.0)), GraphChoice::Eager);
        assert_eq!(resolve("something-else", Some(4.0)), GraphChoice::Capture);
    }

    #[test]
    fn choice_flags() {
        assert_eq!(GraphChoice::Capture.flag(), Some("graph-captured"));
        assert_eq!(GraphChoice::Eager.flag(), Some("graph-eager"));
        assert_eq!(GraphChoice::Unknown.flag(), None);
    }
}
