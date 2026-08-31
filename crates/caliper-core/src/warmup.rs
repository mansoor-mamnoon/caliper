//! Steady-state (warm-up) detection for a stream of timing samples.
//!
//! A GPU's first iterations of a kernel are slow: the clocks are ramping and the
//! JIT / caches are cold. Benchmark helpers that use a small fixed warm-up (the
//! common default is 25 iterations) can start measuring while the series is
//! still 10-30% above its steady value. This module finds the index from which
//! the samples are actually warm, so the caller can discard the rest.
//!
//! Method: cold samples are *high* and the series settles *down* to a floor.
//! Take the median of the final `window` samples as the steady reference, then
//! walk forward and return the first index whose trailing window has a median
//! within `tol` (relative) of that reference.

use serde::{Deserialize, Serialize};

/// Options for [`steady_state`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WarmupOpts {
    /// Size of the trailing window whose median is compared to the reference.
    pub window: usize,
    /// Relative tolerance: a window is "warm" once its median is within
    /// `reference * (1 + tol)`.
    pub tol: f64,
    /// Require at least this many samples to remain after the warm-up point;
    /// if convergence happens later than that, back the start off so this many
    /// warm samples are kept.
    pub min_warm: usize,
}

impl Default for WarmupOpts {
    fn default() -> Self {
        Self {
            window: 20,
            tol: 0.02,
            min_warm: 30,
        }
    }
}

/// The result of [`steady_state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Warmup {
    /// Index of the first warm sample; `times[start..]` are the samples to keep.
    pub start: usize,
    /// Whether a stable window within tolerance was actually found (vs. the
    /// series never settling, in which case `start` is a best-effort fallback).
    pub converged: bool,
}

/// How a caller wants warm-up handled: auto-detect the steady state, or trim a
/// fixed number of leading samples.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WarmupPlan {
    /// Trim exactly this many leading samples and skip detection. `None` means
    /// auto-detect with [`WarmupPlan::opts`].
    pub fixed: Option<usize>,
    /// Options for auto steady-state detection (ignored when `fixed` is set).
    pub opts: WarmupOpts,
}

impl WarmupPlan {
    /// Auto steady-state detection with the given options.
    #[must_use]
    pub fn auto(opts: WarmupOpts) -> Self {
        Self { fixed: None, opts }
    }

    /// Trim a fixed `n` leading samples.
    #[must_use]
    pub fn fixed(n: usize) -> Self {
        Self {
            fixed: Some(n),
            opts: WarmupOpts::default(),
        }
    }

    /// Resolve this plan against a concrete series. A fixed warm-up is clamped so
    /// at least one sample always remains.
    #[must_use]
    pub fn resolve(&self, times: &[f64]) -> Warmup {
        match self.fixed {
            Some(n) => Warmup {
                start: n.min(times.len().saturating_sub(1)),
                converged: true,
            },
            None => steady_state(times, self.opts),
        }
    }
}

/// Find the first warm sample index for `times`.
///
/// Returns `Warmup { start: 0, converged: true }` for an empty or very short
/// series (nothing to trim). Never returns a `start` that would leave fewer
/// than `opts.min_warm` samples unless the whole series is shorter than that.
#[must_use]
pub fn steady_state(times: &[f64], opts: WarmupOpts) -> Warmup {
    let n = times.len();
    let window = opts.window.max(1);

    if n <= window || n <= opts.min_warm {
        return Warmup {
            start: 0,
            converged: true,
        };
    }

    let reference = window_median(&times[n - window..]);
    if reference <= 0.0 || !reference.is_finite() {
        return Warmup {
            start: 0,
            converged: false,
        };
    }
    let ceiling = reference * (1.0 + opts.tol);

    let last_possible_start = n - opts.min_warm;
    for start in 0..=last_possible_start {
        let end = (start + window).min(n);
        if window_median(&times[start..end]) <= ceiling {
            return Warmup {
                start,
                converged: true,
            };
        }
    }

    // Never settled within tolerance: keep the last `min_warm` samples.
    Warmup {
        start: last_possible_start,
        converged: false,
    }
}

fn window_median(win: &[f64]) -> f64 {
    let mut v: Vec<f64> = win.to_vec();
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(window: usize, tol: f64, min_warm: usize) -> WarmupOpts {
        WarmupOpts {
            window,
            tol,
            min_warm,
        }
    }

    /// A cold ramp that decays to `steady` with time-constant `tau`, then holds.
    fn ramp(steady: f64, start_hot: f64, tau: f64, ramp_len: usize, tail: usize) -> Vec<f64> {
        let mut v = Vec::with_capacity(ramp_len + tail);
        for i in 0..ramp_len {
            v.push(steady + (start_hot - steady) * (-(i as f64) / tau).exp());
        }
        v.extend(std::iter::repeat_n(steady, tail));
        v
    }

    #[test]
    fn flat_series_is_warm_from_the_start() {
        let w = steady_state(&[50.0; 100], WarmupOpts::default());
        assert_eq!(
            w,
            Warmup {
                start: 0,
                converged: true
            }
        );
    }

    #[test]
    fn short_series_is_not_trimmed() {
        let w = steady_state(&[100.0, 90.0, 80.0], WarmupOpts::default());
        assert_eq!(w.start, 0);
        assert!(w.converged);
    }

    #[test]
    fn noisy_but_stationary_series_is_warm_early() {
        let xs: Vec<f64> = (0..120)
            .map(|i| 50.0 + if i % 2 == 0 { 0.3 } else { -0.3 })
            .collect();
        let w = steady_state(&xs, opts(20, 0.02, 30));
        assert!(w.converged);
        assert!(w.start <= 2, "start was {}", w.start);
    }

    #[test]
    fn single_spike_does_not_delay_warm_up() {
        let mut xs = vec![50.0; 120];
        xs[7] = 250.0; // one outlier; window median shrugs it off
        let w = steady_state(&xs, opts(20, 0.02, 30));
        assert!(w.converged);
        assert!(w.start <= 1, "start was {}", w.start);
    }

    #[test]
    fn cold_ramp_warm_up_is_after_the_ramp() {
        // Hot start 90us, steady 50us, decays over ~15 samples, then 60 flat.
        let xs = ramp(50.0, 90.0, 6.0, 40, 60);
        let w = steady_state(&xs, opts(20, 0.02, 30));
        assert!(w.converged);
        // within tolerance somewhere in the ramp's tail, well before the flat part
        assert!((10..40).contains(&w.start), "start was {}", w.start);
    }

    #[test]
    fn slow_ramp_shows_why_a_fixed_25_is_wrong() {
        // Still ~14% above steady at sample 25; converges near ~140.
        let xs = ramp(50.0, 120.0, 40.0, 200, 40);
        assert!(xs[25] > 50.0 * 1.10, "sample 25 was {}", xs[25]);

        let w = steady_state(&xs, opts(20, 0.02, 30));
        assert!(w.converged);
        assert!(
            w.start > 25 && w.start < 210,
            "auto warm-up start was {} (a fixed warmup=25 would still be on cold clocks)",
            w.start
        );
    }

    #[test]
    fn warmup_plan_fixed_trims_exactly_n_and_is_clamped() {
        let xs = vec![9.0; 50];
        assert_eq!(
            WarmupPlan::fixed(25).resolve(&xs),
            Warmup {
                start: 25,
                converged: true
            }
        );
        // clamped so at least one sample survives
        assert_eq!(WarmupPlan::fixed(999).resolve(&xs).start, 49);
    }

    #[test]
    fn warmup_plan_auto_matches_steady_state() {
        let xs = ramp(50.0, 120.0, 40.0, 200, 40);
        assert_eq!(
            WarmupPlan::auto(WarmupOpts::default()).resolve(&xs),
            steady_state(&xs, WarmupOpts::default())
        );
    }

    #[test]
    fn never_settles_falls_back_to_last_min_warm_samples() {
        // Monotonic decay that never reaches within tol of its own tail median.
        let xs: Vec<f64> = (0..200).map(|i| 1000.0 / (1.0 + i as f64 * 0.01)).collect();
        let w = steady_state(&xs, opts(10, 0.001, 25));
        assert!(!w.converged);
        assert_eq!(w.start, xs.len() - 25);
    }
}
