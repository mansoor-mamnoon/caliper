//! Summary statistics for a set of timing samples.
//!
//! caliper reports a *distribution*, not a single mean: the p10 / p50 / p90 of
//! the samples plus a robust spread (median absolute deviation). This module is
//! the one place those numbers are computed, so every command agrees on method.
//!
//! Percentiles use linear interpolation between the two nearest ranks -- the
//! same convention as NumPy's default (`method="linear"`, i.e. Hyndman & Fan
//! type 7). The coefficient of variation uses the sample (Bessel-corrected)
//! standard deviation.

use serde::{Deserialize, Serialize};

/// A distribution summary of timing samples, in whatever unit the samples were.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    /// Number of samples summarised.
    pub n: usize,
    /// Smallest sample.
    pub min: f64,
    /// 10th percentile.
    pub p10: f64,
    /// 50th percentile (median).
    pub p50: f64,
    /// 90th percentile.
    pub p90: f64,
    /// Largest sample.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Median absolute deviation: `median(|x - median(x)|)`. Raw, unscaled;
    /// multiply by 1.4826 for a normal-consistent estimate of the standard
    /// deviation.
    pub mad: f64,
    /// Coefficient of variation: sample standard deviation divided by the mean.
    /// `None` when the mean is zero.
    pub cov: Option<f64>,
}

/// Compute the `q`-quantile (`q` in `0.0..=1.0`) of an already-sorted,
/// non-empty slice, using linear interpolation between the two nearest ranks.
///
/// # Panics
/// Panics if `sorted` is empty or `q` is outside `0.0..=1.0`.
#[must_use]
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    assert!(!sorted.is_empty(), "quantile of an empty slice");
    assert!((0.0..=1.0).contains(&q), "quantile q must be in 0.0..=1.0");
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = q * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// The median of an already-sorted, non-empty slice.
///
/// # Panics
/// Panics if `sorted` is empty.
#[must_use]
pub fn median_sorted(sorted: &[f64]) -> f64 {
    quantile_sorted(sorted, 0.5)
}

/// Summarise timing samples.
///
/// Returns `None` if `samples` is empty or contains a non-finite value -- the
/// measurement layer is expected to hand over clean, finite data, and a silent
/// NaN would corrupt every downstream number.
#[must_use]
pub fn summarize(samples: &[f64]) -> Option<Summary> {
    if samples.is_empty() || samples.iter().any(|x| !x.is_finite()) {
        return None;
    }

    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(f64::total_cmp);

    let n = sorted.len();
    let median = median_sorted(&sorted);

    let mut abs_dev: Vec<f64> = sorted.iter().map(|x| (x - median).abs()).collect();
    abs_dev.sort_by(f64::total_cmp);
    let mad = median_sorted(&abs_dev);

    let mean = sorted.iter().sum::<f64>() / n as f64;
    let cov = if mean == 0.0 {
        None
    } else {
        let var = if n < 2 {
            0.0
        } else {
            sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        };
        Some(var.sqrt() / mean)
    };

    Some(Summary {
        n,
        min: sorted[0],
        p10: quantile_sorted(&sorted, 0.10),
        p50: median,
        p90: quantile_sorted(&sorted, 0.90),
        max: sorted[n - 1],
        mean,
        mad,
        cov,
    })
}

/// Coefficient of variation of the per-pass medians, used to judge whether a
/// measurement reproduces across independent passes. Returns `None` with fewer
/// than two passes, or if the mean of the medians is zero.
#[must_use]
pub fn cross_pass_cov(pass_medians: &[f64]) -> Option<f64> {
    if pass_medians.len() < 2 || pass_medians.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mean = pass_medians.iter().sum::<f64>() / pass_medians.len() as f64;
    if mean == 0.0 {
        return None;
    }
    let var = pass_medians.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (pass_medians.len() - 1) as f64;
    Some(var.sqrt() / mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        let tol = 1e-9 * b.abs().max(1.0);
        assert!((a - b).abs() <= tol, "expected {b}, got {a}");
    }

    #[test]
    fn quantiles_of_one_to_ten_match_numpy_linear() {
        let xs: Vec<f64> = (1..=10).map(f64::from).collect();
        close(quantile_sorted(&xs, 0.0), 1.0);
        close(quantile_sorted(&xs, 0.10), 1.9);
        close(quantile_sorted(&xs, 0.50), 5.5);
        close(quantile_sorted(&xs, 0.90), 9.1);
        close(quantile_sorted(&xs, 1.0), 10.0);
    }

    #[test]
    fn summary_of_one_to_ten_is_hand_computed() {
        let xs: Vec<f64> = (1..=10).map(f64::from).collect();
        let s = summarize(&xs).unwrap();
        assert_eq!(s.n, 10);
        close(s.min, 1.0);
        close(s.max, 10.0);
        close(s.p10, 1.9);
        close(s.p50, 5.5);
        close(s.p90, 9.1);
        close(s.mean, 5.5);
        close(s.mad, 2.5); // median of |x-5.5| = median of {0.5,0.5,1.5,1.5,2.5,2.5,3.5,3.5,4.5,4.5}
        close(s.cov.unwrap(), (82.5f64 / 9.0).sqrt() / 5.5);
    }

    #[test]
    fn summary_is_order_independent() {
        let a = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let mut b = a;
        b.reverse();
        assert_eq!(summarize(&a), summarize(&b));
    }

    #[test]
    fn percentiles_are_ordered() {
        for seed in 0u64..64 {
            // cheap deterministic pseudo-random sample
            let xs: Vec<f64> = (0..50)
                .map(|i| {
                    let x = (seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(i * 2654435761)) as f64;
                    (x % 1000.0).abs() + 1.0
                })
                .collect();
            let s = summarize(&xs).unwrap();
            assert!(s.min <= s.p10 && s.p10 <= s.p50 && s.p50 <= s.p90 && s.p90 <= s.max);
        }
    }

    #[test]
    fn constant_input_has_zero_spread() {
        let s = summarize(&[7.0; 20]).unwrap();
        close(s.p10, 7.0);
        close(s.p50, 7.0);
        close(s.p90, 7.0);
        close(s.mad, 0.0);
        close(s.cov.unwrap(), 0.0);
    }

    #[test]
    fn single_sample_summary() {
        let s = summarize(&[42.0]).unwrap();
        assert_eq!(s.n, 1);
        close(s.p50, 42.0);
        close(s.mad, 0.0);
        close(s.cov.unwrap(), 0.0);
    }

    #[test]
    fn empty_and_non_finite_are_rejected() {
        assert!(summarize(&[]).is_none());
        assert!(summarize(&[1.0, f64::NAN, 3.0]).is_none());
        assert!(summarize(&[1.0, f64::INFINITY]).is_none());
    }

    #[test]
    fn cross_pass_cov_needs_two_finite_passes() {
        assert!(cross_pass_cov(&[]).is_none());
        assert!(cross_pass_cov(&[5.0]).is_none());
        assert!(cross_pass_cov(&[5.0, f64::NAN]).is_none());
        close(cross_pass_cov(&[10.0, 10.0, 10.0]).unwrap(), 0.0);
        let cov = cross_pass_cov(&[100.0, 102.0, 98.0]).unwrap();
        assert!(cov > 0.0 && cov < 0.05);
    }
}
