//! The on-device oracle checks.
//!
//! Each oracle is a kernel whose *true* behaviour is knowable from first
//! principles, so it pins one measurement path without trusting caliper. This
//! module holds the analytic expectations and the pass/fail checks; the kernels
//! themselves are CUDA C++ in `caliper-gpu/kernels/` and are compiled and run on
//! a CUDA host. The check functions here are pure and `cargo test`-covered.
//!
//! | Oracle | Pins | Expectation |
//! |--------|------|-------------|
//! | O1 `busy(target_ns)` | the timing path | `p50_us == target_ns / 1000` |
//! | O2 `triad(n)` | GB/s math + L2 flush | `gbps == 3 * bytes / (p50_us * 1e3)` |
//! | O3 `fma_peak` | TFLOP/s math + compute-bound branch | `>= 90%` of the FMA peak |
//! | O4 `one_op` | launch overhead | `p50` ≈ the launch gap; graph replay `< 1 us` |
//! | O6 `throttle_bait` | throttle detection | samples dropped, reasons reported |

/// The result of one oracle check.
#[derive(Debug, Clone, PartialEq)]
pub struct OracleCheck {
    /// Short name, e.g. `"o1_point"`.
    pub name: String,
    /// Whether the measurement met its expectation.
    pub passed: bool,
    /// What was measured.
    pub measured: f64,
    /// What first principles say it should be.
    pub expected: f64,
    /// Relative tolerance applied (0.0 for boolean checks).
    pub tolerance: f64,
    /// Human-readable explanation.
    pub detail: String,
}

impl OracleCheck {
    fn within(name: &str, measured: f64, expected: f64, tol: f64) -> Self {
        let rel = if expected == 0.0 {
            measured.abs()
        } else {
            (measured - expected).abs() / expected.abs()
        };
        let passed = rel <= tol;
        Self {
            name: name.to_string(),
            passed,
            measured,
            expected,
            tolerance: tol,
            detail: format!(
                "measured {measured:.4}, expected {expected:.4} (rel err {:.2}%, tol {:.1}%)",
                rel * 100.0,
                tol * 100.0
            ),
        }
    }

    fn boolean(name: &str, passed: bool, measured: f64, detail: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            passed,
            measured,
            expected: 1.0,
            tolerance: 0.0,
            detail: detail.into(),
        }
    }
}

/// A least-squares line fit, `y = slope * x + intercept`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineFit {
    /// Fitted slope.
    pub slope: f64,
    /// Fitted intercept.
    pub intercept: f64,
    /// Coefficient of determination.
    pub r2: f64,
}

/// Ordinary least-squares fit of `points` (`(x, y)`).
///
/// Returns `None` for fewer than two points, or if every `x` is identical.
#[must_use]
pub fn fit_line(points: &[(f64, f64)]) -> Option<LineFit> {
    let n = points.len() as f64;
    if points.len() < 2 {
        return None;
    }
    let (sx, sy) = points
        .iter()
        .fold((0.0, 0.0), |(ax, ay), &(x, y)| (ax + x, ay + y));
    let (mx, my) = (sx / n, sy / n);
    let sxx: f64 = points.iter().map(|&(x, _)| (x - mx).powi(2)).sum();
    let sxy: f64 = points.iter().map(|&(x, y)| (x - mx) * (y - my)).sum();
    if sxx == 0.0 {
        return None;
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let ss_tot: f64 = points.iter().map(|&(_, y)| (y - my).powi(2)).sum();
    let ss_res: f64 = points
        .iter()
        .map(|&(x, y)| (y - (slope * x + intercept)).powi(2))
        .sum();
    let r2 = if ss_tot == 0.0 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    };
    Some(LineFit {
        slope,
        intercept,
        r2,
    })
}

// --- O1: calibrated duration ------------------------------------------------

/// Per-launch time a `busy(target_ns)` kernel should show, in microseconds.
#[must_use]
pub fn o1_expected_us(target_ns: f64) -> f64 {
    target_ns / 1000.0
}

/// Check one point of the O1 sweep. Tolerance is 3% at or above 50 us, 10% below
/// (short kernels are noisier).
#[must_use]
pub fn check_o1_point(target_ns: f64, measured_us: f64) -> OracleCheck {
    let expected = o1_expected_us(target_ns);
    let tol = if expected >= 50.0 { 0.03 } else { 0.10 };
    OracleCheck::within("o1_point", measured_us, expected, tol)
}

/// Check that a full O1 sweep is linear with unit slope. `points` are
/// `(expected_us, measured_us)`.
///
/// Only the slope gates the pass/fail (Appendix A: `slope ∈ [0.97, 1.03]`). The
/// fitted intercept is reported in `detail` — it should be close to the launch
/// overhead, but pinning that number is [`check_o4_launch_overhead`]'s job.
#[must_use]
pub fn check_o1_linearity(points: &[(f64, f64)]) -> OracleCheck {
    match fit_line(points) {
        Some(fit) => {
            let passed = (0.97..=1.03).contains(&fit.slope);
            OracleCheck {
                name: "o1_linearity".to_string(),
                passed,
                measured: fit.slope,
                expected: 1.0,
                tolerance: 0.03,
                detail: format!(
                    "slope {:.4} (want 0.97..=1.03), intercept {:.3} us, R^2 {:.5}",
                    fit.slope, fit.intercept, fit.r2
                ),
            }
        }
        None => OracleCheck::boolean("o1_linearity", false, 0.0, "need >= 2 distinct points"),
    }
}

// --- O2: streaming triad --------------------------------------------------

/// Achieved bandwidth (GB/s) for a triad over `bytes_per_array` bytes measured
/// at `p50_us` per launch. The triad touches three arrays (read b, read c,
/// write a).
#[must_use]
pub fn o2_achieved_gbps(bytes_per_array: f64, p50_us: f64) -> f64 {
    3.0 * bytes_per_array / (p50_us * 1.0e3)
}

/// Check O2 bandwidth against an independent reference (e.g. an `nvbandwidth`
/// device-to-device number), 5% tolerance.
#[must_use]
pub fn check_o2_bandwidth(bytes_per_array: f64, p50_us: f64, reference_gbps: f64) -> OracleCheck {
    OracleCheck::within(
        "o2_bandwidth",
        o2_achieved_gbps(bytes_per_array, p50_us),
        reference_gbps,
        0.05,
    )
}

/// Check the L2-flush A/B: with the transfer smaller than L2, not flushing
/// should be at least 2x faster (cache hits); at 4x L2 the two should agree
/// within 5%.
#[must_use]
pub fn check_o2_flush_ab(
    gbps_flush_on: f64,
    gbps_flush_off: f64,
    bytes_per_array: f64,
    l2_bytes: f64,
) -> OracleCheck {
    let ratio = gbps_flush_off / gbps_flush_on;
    if bytes_per_array < l2_bytes {
        OracleCheck {
            name: "o2_flush_ab_small".to_string(),
            passed: ratio >= 2.0,
            measured: ratio,
            expected: 2.0,
            tolerance: 0.0,
            detail: format!("flush-off/flush-on = {ratio:.2}x (want >= 2x when it fits in L2)"),
        }
    } else if bytes_per_array >= 4.0 * l2_bytes {
        OracleCheck {
            name: "o2_flush_ab_large".to_string(),
            passed: (ratio - 1.0).abs() < 0.05,
            measured: ratio,
            expected: 1.0,
            tolerance: 0.05,
            detail: format!("flush-off/flush-on = {ratio:.3}x (want ~1x well past L2)"),
        }
    } else {
        OracleCheck::boolean(
            "o2_flush_ab",
            true,
            ratio,
            "transfer is between L2 and 4x L2; A/B not asserted",
        )
    }
}

// --- O3: FMA peak -------------------------------------------------------------

/// Flops a `fma_peak` launch performs: 2 per FMA, `ilp` independent lanes.
#[must_use]
pub fn o3_flops(total_threads: f64, iters: f64, ilp: f64) -> f64 {
    2.0 * total_threads * iters * ilp
}

/// Achieved TFLOP/s for an O3 launch.
#[must_use]
pub fn o3_achieved_tflops(total_threads: f64, iters: f64, ilp: f64, p50_us: f64) -> f64 {
    o3_flops(total_threads, iters, ilp) / (p50_us * 1.0e6)
}

/// Check O3 reaches at least 90% of the documented FMA peak and is classified
/// compute-bound.
#[must_use]
pub fn check_o3_fma(
    total_threads: f64,
    iters: f64,
    ilp: f64,
    p50_us: f64,
    documented_peak_tflops: f64,
    bound: &str,
) -> OracleCheck {
    let achieved = o3_achieved_tflops(total_threads, iters, ilp, p50_us);
    let frac = achieved / documented_peak_tflops;
    OracleCheck {
        name: "o3_fma".to_string(),
        passed: frac >= 0.90 && bound == "compute",
        measured: achieved,
        expected: documented_peak_tflops,
        tolerance: 0.10,
        detail: format!(
            "{achieved:.1} TFLOP/s = {:.1}% of peak, bound = {bound:?} (want >= 90% and compute)",
            frac * 100.0
        ),
    }
}

// --- O4: launch overhead --------------------------------------------------

/// Check O4: the eager per-launch time matches the `nsys` launch gap within 20%,
/// and graph replay drops per-launch cost below 1 us.
#[must_use]
pub fn check_o4_launch_overhead(eager_us: f64, graph_us: f64, nsys_gap_us: f64) -> OracleCheck {
    let rel = (eager_us - nsys_gap_us).abs() / nsys_gap_us.max(f64::MIN_POSITIVE);
    OracleCheck {
        name: "o4_launch_overhead".to_string(),
        passed: rel <= 0.20 && graph_us < 1.0,
        measured: eager_us,
        expected: nsys_gap_us,
        tolerance: 0.20,
        detail: format!(
            "eager {eager_us:.2} us vs nsys gap {nsys_gap_us:.2} us ({:.1}%); graph {graph_us:.2} us (want < 1)",
            rel * 100.0
        ),
    }
}

// --- O6: throttle detection ---------------------------------------------------

/// Check O6: inducing throttle drops samples and reports a power/thermal reason
/// (Appendix A: `SW_POWER_CAP` or an `HW_*` / thermal token).
#[must_use]
pub fn check_o6_throttle(invalidated_samples: u64, throttle_reasons: &[String]) -> OracleCheck {
    let power_or_thermal = throttle_reasons.iter().any(|r| {
        let u = r.to_uppercase();
        u.contains("POWER") || u.contains("THERMAL") || u.starts_with("HW_")
    });
    let passed = invalidated_samples > 0 && power_or_thermal;
    OracleCheck::boolean(
        "o6_throttle",
        passed,
        invalidated_samples as f64,
        format!(
            "{invalidated_samples} sample(s) dropped, reasons {throttle_reasons:?} \
             (want > 0 and a power/thermal reason)"
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-6 * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn o1_expectation_is_target_over_1000() {
        approx(o1_expected_us(200_000.0), 200.0);
        approx(o1_expected_us(5_000.0), 5.0);
    }

    #[test]
    fn o1_point_check_uses_a_looser_tolerance_below_50us() {
        assert!(check_o1_point(200_000.0, 205.0).passed); // 2.5% at 200us -> ok
        assert!(!check_o1_point(200_000.0, 210.0).passed); // 5% at 200us -> fail
        assert!(check_o1_point(5_000.0, 5.4).passed); // 8% at 5us -> ok
        assert!(!check_o1_point(5_000.0, 5.7).passed); // 14% at 5us -> fail
    }

    #[test]
    fn line_fit_recovers_slope_and_intercept() {
        let pts: Vec<(f64, f64)> = (0..10).map(|i| (i as f64, 2.0 * i as f64 + 3.0)).collect();
        let fit = fit_line(&pts).unwrap();
        approx(fit.slope, 2.0);
        approx(fit.intercept, 3.0);
        approx(fit.r2, 1.0);
        assert!(fit_line(&[(1.0, 1.0)]).is_none());
        assert!(fit_line(&[(1.0, 1.0), (1.0, 2.0)]).is_none());
    }

    #[test]
    fn o1_linearity_accepts_unit_slope_rejects_drift() {
        let good: Vec<(f64, f64)> = [1.0, 5.0, 50.0, 200.0, 1000.0]
            .iter()
            .map(|&x| (x, 1.006 * x + 6.0))
            .collect();
        assert!(check_o1_linearity(&good).passed);

        let bad: Vec<(f64, f64)> = [1.0, 5.0, 50.0, 200.0, 1000.0]
            .iter()
            .map(|&x| (x, 1.15 * x))
            .collect();
        assert!(!check_o1_linearity(&bad).passed);
    }

    #[test]
    fn o2_bandwidth_math_and_check() {
        // 1 GiB per array, 3 GiB moved in 3.3 ms -> ~976 GB/s
        let gbps = o2_achieved_gbps(1024.0 * 1024.0 * 1024.0, 3300.0);
        approx(gbps, 3.0 * 1024.0 * 1024.0 * 1024.0 / 3.3e6);
        assert!(check_o2_bandwidth(1024.0 * 1024.0 * 1024.0, 3300.0, gbps * 1.02).passed);
        assert!(!check_o2_bandwidth(1024.0 * 1024.0 * 1024.0, 3300.0, gbps * 1.2).passed);
    }

    #[test]
    fn o2_flush_ab_small_and_large() {
        let l2 = 72.0 * 1024.0 * 1024.0;
        assert!(check_o2_flush_ab(900.0, 3600.0, l2 / 2.0, l2).passed); // 4x when it fits
        assert!(!check_o2_flush_ab(900.0, 1200.0, l2 / 2.0, l2).passed); // only 1.3x
        assert!(check_o2_flush_ab(900.0, 909.0, 5.0 * l2, l2).passed); // ~1x past L2
        assert!(!check_o2_flush_ab(900.0, 1100.0, 5.0 * l2, l2).passed);
    }

    #[test]
    fn o3_fma_needs_90pct_and_compute_bound() {
        let threads = 128.0 * 128.0 * 256.0;
        let iters = 100_000.0;
        let ilp = 4.0;
        let tflops = o3_achieved_tflops(threads, iters, ilp, 1000.0);
        assert!(check_o3_fma(threads, iters, ilp, 1000.0, tflops * 1.05, "compute").passed);
        assert!(!check_o3_fma(threads, iters, ilp, 1000.0, tflops * 1.20, "compute").passed);
        assert!(!check_o3_fma(threads, iters, ilp, 1000.0, tflops, "memory").passed);
    }

    #[test]
    fn o4_needs_matching_eager_gap_and_tiny_graph() {
        assert!(check_o4_launch_overhead(6.0, 0.4, 5.5).passed);
        assert!(!check_o4_launch_overhead(9.0, 0.4, 5.5).passed); // eager 64% off
        assert!(!check_o4_launch_overhead(6.0, 1.5, 5.5).passed); // graph not eliminated
    }

    #[test]
    fn o6_needs_dropped_samples_and_a_power_or_thermal_reason() {
        assert!(check_o6_throttle(214, &["SW_POWER_CAP".to_string()]).passed);
        assert!(check_o6_throttle(5, &["HW_THERMAL_SLOWDOWN".to_string()]).passed);
        assert!(!check_o6_throttle(0, &["SW_POWER_CAP".to_string()]).passed); // nothing dropped
        assert!(!check_o6_throttle(10, &[]).passed); // no reason
        assert!(!check_o6_throttle(10, &["GpuIdle".to_string()]).passed); // benign reason
    }
}
